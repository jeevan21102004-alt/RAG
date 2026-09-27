"""Query-aware configuration evaluation (Phase 6B pilot).

Compares GLOBAL vs QUERY-TYPE-SPECIFIC retrieval configurations using
ONLY the existing retrieval-only infrastructure:

- questions: ``data/enterprise_retrieval_questions.json``
- classification: Phase 6A ``query_classifier`` (unmodified)
- retrieval: ``RetrievalBenchmarkAdapter`` with ``enable_generation=False``
- scoring: ``run_evaluation`` + ``calculate_objective`` (unmodified)
- configs: deterministic subset of ``SearchSpace.adaptive_pilot`` (175)

ZERO external calls: no LLM, no Gemini, no network, no RL training.

Pilot budget (labelled, deterministic): 4 configurations x 20 questions
= 80 retrieval-only evaluations. ``run_pilot`` accepts smaller
``configurations`` / ``questions`` overrides so tests stay fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"

PILOT_LABEL = "Phase 6B query-aware pilot (4 configs x 20 questions)"

# Deterministic pilot subset: corners/centre of the 175-config space.
PILOT_CONFIGURATIONS: tuple[dict[str, int], ...] = (
    {"chunk_size": 100, "chunk_overlap": 20, "top_k": 3},
    {"chunk_size": 300, "chunk_overlap": 50, "top_k": 5},
    {"chunk_size": 500, "chunk_overlap": 10, "top_k": 2},
    {"chunk_size": 200, "chunk_overlap": 40, "top_k": 4},
)

VALID_QUERY_TYPES: tuple[str, ...] = (
    "SIMPLE",
    "MULTI_DOCUMENT",
    "CROSS_DOMAIN",
    "TECHNICAL",
)


def pilot_configurations() -> list[dict[str, int]]:
    """Return the deterministic pilot configuration subset."""
    return [dict(c) for c in PILOT_CONFIGURATIONS]


def load_questions() -> list[dict[str, Any]]:
    """Load the 20 enterprise benchmark questions (local JSON)."""
    import json

    path = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def classify_questions(
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Classify every question; returns {case_id: QueryType value}."""
    from .query_classifier import classify_benchmark_question

    items = load_questions() if questions is None else list(questions)
    return {
        str(item.get("case_id", "")): classify_benchmark_question(item).value
        for item in items
    }


def group_by_query_type(
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group question dicts by their deterministic QueryType value."""
    items = load_questions() if questions is None else [dict(q) for q in questions]
    groups: dict[str, list[dict[str, Any]]] = {}
    labels = classify_questions(items)
    for item in items:
        label = labels[str(item.get("case_id", ""))]
        groups.setdefault(label, []).append(item)
    return groups


def evaluate_configuration(
    configuration: Mapping[str, int],
    questions: Sequence[Mapping[str, Any]] | None = None,
    experiment_id: str = "query-aware-eval",
) -> dict[str, Any]:
    """Evaluate one config over questions using retrieval-only infra."""
    from .evaluation_runner import run_evaluation
    from .evaluation_schema import EvaluationCase
    from .experiment_config import ExperimentConfig
    from .retrieval_benchmark_adapter import RetrievalBenchmarkAdapter

    items = load_questions() if questions is None else [dict(q) for q in questions]
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name="Query-aware configuration evaluation",
        description="Retrieval-only evaluation (generation disabled).",
        parameters=dict(configuration),
    )
    adapter = RetrievalBenchmarkAdapter(
        enable_generation=False, data_dir=ENTERPRISE_KB_DIR)
    evaluations: list[dict[str, Any]] = []
    for item in items:
        case = EvaluationCase(
            case_id=str(item.get("case_id", "")),
            question=str(item.get("question", "")),
            category=str(item.get("category", "")),
            expected_answer=item.get("expected_answer"),
            expected_action=item.get("expected_action"),
            relevant_documents=list(item.get("relevant_documents", []) or []),
        )
        response = adapter.run(case, config)
        result = run_evaluation(case, response)
        metrics = result.metadata.get("retrieval_metrics", {}) or {}
        latency = response.metadata.get("retrieval_latency_ms")
        if latency is None:
            latency = response.metadata.get("total_latency_ms")
        if latency is None:
            latency = response.latency_ms
        evaluations.append({
            "case_id": case.case_id,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "context_relevance": result.metadata.get("context_relevance"),
            "latency_ms": latency,
            "status": result.status,
        })
    return {
        "configuration": dict(configuration),
        "evaluations": evaluations,
        "aggregated": aggregate_metrics(evaluations),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def aggregate_metrics(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average retrieval metrics; objective via existing calculate_objective."""
    from .objective import calculate_objective

    f1 = _mean([e.get("f1") for e in evaluations])
    ctx = _mean([e.get("context_relevance") for e in evaluations])
    lat = _mean([e.get("latency_ms") for e in evaluations])
    obj = calculate_objective(f1, ctx, lat)
    return {
        "n": len(list(evaluations)),
        "precision": _mean([e.get("precision") for e in evaluations]),
        "recall": _mean([e.get("recall") for e in evaluations]),
        "f1": f1,
        "context_relevance": ctx,
        "latency_ms": lat,
        "objective_score": obj.objective_score,
        "missing_components": list(obj.missing_components),
    }


def _config_key(configuration: Mapping[str, int]) -> tuple:
    return (int(configuration.get("chunk_size", 0)),
            int(configuration.get("chunk_overlap", 0)),
            int(configuration.get("top_k", 0)))


def select_best(results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Deterministically pick highest objective (ties -> smallest params)."""
    scored = [r for r in results
              if (r.get("aggregated", {}) or {}).get("objective_score") is not None]
    if not scored:
        return None
    scored = sorted(scored, key=lambda r: (
        -float(r["aggregated"]["objective_score"]), _config_key(r["configuration"])))
    return scored[0]


def compare_global_vs_query_specific(
    per_type_best: Mapping[str, Mapping[str, Any] | None],
    global_best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare global best vs per-type bests (measured values only)."""
    out: dict[str, Any] = {"global": None, "per_type": {}, "difference": {}}
    if global_best is not None:
        out["global"] = {"configuration": dict(global_best["configuration"]),
                         "aggregated": dict(global_best["aggregated"])}
    for qtype, best in per_type_best.items():
        if best is None:
            out["per_type"][qtype] = None
            continue
        out["per_type"][qtype] = {"configuration": dict(best["configuration"]),
                                  "aggregated": dict(best["aggregated"])}
    if global_best is not None:
        g = float(global_best["aggregated"]["objective_score"])
        for qtype, best in per_type_best.items():
            if best is None:
                out["difference"][qtype] = None
            else:
                out["difference"][qtype] = (
                    float(best["aggregated"]["objective_score"]) - g)
    return out

def run_pilot(
    configurations: Sequence[Mapping[str, int]] | None = None,
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the deterministic query-aware pilot (retrieval-only, local).

    Default budget: 4 configurations x 20 questions = 80 evaluations.
    Smaller overrides keep unit tests fast. Returns measured values only.
    """
    configs = (
        pilot_configurations()
        if configurations is None
        else [dict(c) for c in configurations]
    )
    items = load_questions() if questions is None else [dict(q) for q in questions]
    groups = group_by_query_type(items)
    counts = {qtype: len(qitems) for qtype, qitems in groups.items()}

    config_results = [evaluate_configuration(c, items) for c in configs]
    total_evals = sum(len(r["evaluations"]) for r in config_results)
    global_best = select_best(config_results)

    per_type_best: dict[str, dict[str, Any] | None] = {}
    for qtype, qitems in groups.items():
        ids = {str(q.get("case_id", "")) for q in qitems}
        scoped = []
        for result in config_results:
            subset = [e for e in result["evaluations"] if e["case_id"] in ids]
            scoped.append({
                "configuration": dict(result["configuration"]),
                "aggregated": aggregate_metrics(subset),
                "evaluations": subset,
            })
        per_type_best[qtype] = select_best(scoped)

    return {
        "label": PILOT_LABEL,
        "budget": {
            "configurations": len(configs),
            "questions": len(items),
            "retrieval_evaluations": total_evals,
        },
        "question_counts": counts,
        "global_best": global_best,
        "per_type_best": per_type_best,
        "comparison": compare_global_vs_query_specific(per_type_best, global_best),
        "config_results": config_results,
    }
