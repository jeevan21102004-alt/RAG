"""Dashboard service layer for the AdaptiveRAG demo dashboard (Phase 5).

Thin, API-free layer over existing modules. No evaluation, retrieval,
diagnostic, or optimization logic is duplicated here -- everything
delegates to the established pipeline:

- corpus: ``data/enterprise_kb/``
- questions: ``data/enterprise_retrieval_questions.json``
- split: ``data/enterprise_rl_split.json``
- search space: :class:`SearchSpace.adaptive_pilot` (175 configs)
- evaluation: ``RetrievalBenchmarkAdapter`` (generation disabled) +
  ``run_experiment`` + ``calculate_objective``
- diagnostics: ``diagnose_result`` / ``diagnose_system``

Streamlit is NEVER imported here so business logic stays testable
without a browser.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"
ENTERPRISE_QUESTIONS_PATH = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"
RL_SPLIT_PATH = PROJECT_ROOT / "data" / "enterprise_rl_split.json"

CATEGORIES = ["hr", "engineering", "finance", "product", "security"]

# Deterministic demo subset: first 2 enterprise questions (same as pilots).
DEMO_QUESTION_IDS = ["ent-001", "ent-002"]

# Phase 4 smoke-test comparison is a *historical record* of measured values
# from the Phase 4 report (budget=4, seed=42, test questions). It is labelled
# as such in the dashboard and never presented as live results.
PHASE4_SMOKE_COMPARISON = [
    {"method": "random", "budget": 4, "best_objective": 0.7434,
     "best_f1": 0.8333, "context_relevance": 0.8295,
     "latency_ms": 22.44, "configurations_evaluated": 4},
    {"method": "grid", "budget": 4, "best_objective": 0.7419,
     "best_f1": 0.8333, "context_relevance": 0.8295,
     "latency_ms": 85.88, "configurations_evaluated": 4},
    {"method": "adaptive", "budget": 4, "best_objective": 0.7434,
     "best_f1": 0.8333, "context_relevance": 0.8295,
     "latency_ms": 22.44, "configurations_evaluated": 4},
    {"method": "rl", "budget": 4, "best_objective": 0.6441,
     "best_f1": 0.5833, "context_relevance": 0.8295,
     "latency_ms": 85.92, "configurations_evaluated": 4},
]
PHASE4_SMOKE_LABEL = "Phase 4 smoke-test comparison (budget=4, seed=42)"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_enterprise_documents() -> list[dict[str, Any]]:
    """Load corpus metadata + text. Pure local filesystem read."""
    docs: list[dict[str, Any]] = []
    for category in CATEGORIES:
        folder = ENTERPRISE_KB_DIR / category
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            docs.append({
                "name": path.name,
                "category": category,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "words": len(text.split()),
                "chars": len(text),
                "text": text,
            })
    return docs


def documents_by_category() -> dict[str, int]:
    counts = {c: 0 for c in CATEGORIES}
    for doc in load_enterprise_documents():
        counts[doc["category"]] = counts.get(doc["category"], 0) + 1
    return counts


def load_benchmark_questions() -> list[dict[str, Any]]:
    """Load the 20 enterprise retrieval questions (local JSON)."""
    data = _read_json(ENTERPRISE_QUESTIONS_PATH)
    return list(data)


def load_rl_split() -> dict[str, Any]:
    return _read_json(RL_SPLIT_PATH)


def demo_questions() -> list[dict[str, Any]]:
    by_id = {q["case_id"]: q for q in load_benchmark_questions()}
    return [by_id[cid] for cid in DEMO_QUESTION_IDS if cid in by_id]


def search_space_summary() -> dict[str, Any]:
    from .search_space import SearchSpace

    space = SearchSpace.adaptive_pilot()
    return {
        "name": space.name,
        "parameter_values": {k: list(v) for k, v in space.parameter_values.items()},
        "total_configurations": space.count_combinations(),
    }


def list_configurations(limit: int | None = None) -> list[dict[str, Any]]:
    """Enumerate action_id -> parameters deterministically (no evaluation)."""
    from .search_space import SearchSpace

    space = SearchSpace.adaptive_pilot()
    combos = list(space.generate_parameter_combinations())
    rows = [
        {"action_id": i, "chunk_size": c["chunk_size"],
         "chunk_overlap": c["chunk_overlap"], "top_k": c["top_k"]}
        for i, c in enumerate(combos)
    ]
    return rows[:limit] if limit is not None else rows


def run_retrieval_evaluation(
    question_id: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> dict[str, Any]:
    """Run ONE retrieval-only evaluation locally (generation disabled).

    Reuses ``RetrievalBenchmarkAdapter`` + ``run_evaluation`` directly --
    zero API calls. Returns measured metrics plus expected-vs-retrieved
    document comparison. Never fabricates values.
    """
    from .evaluation_runner import run_evaluation
    from .evaluation_schema import EvaluationCase
    from .experiment_config import ExperimentConfig
    from .retrieval_benchmark_adapter import RetrievalBenchmarkAdapter

    by_id = {q["case_id"]: q for q in load_benchmark_questions()}
    if question_id not in by_id:
        raise ValueError(f"Unknown question: {question_id!r}")
    item = by_id[question_id]
    case = EvaluationCase(**item)
    config = ExperimentConfig(
        experiment_id="dashboard-eval",
        name="Dashboard retrieval evaluation",
        description="Single-question retrieval-only evaluation.",
        parameters={"chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "top_k": top_k},
    )
    adapter = RetrievalBenchmarkAdapter(
        enable_generation=False, data_dir=ENTERPRISE_KB_DIR)
    response = adapter.run(case, config)
    result = run_evaluation(case, response)
    metrics = result.metadata.get("retrieval_metrics", {}) or {}
    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_documents": list(case.relevant_documents),
        "retrieved_documents": list(response.retrieved_documents),
        "retrieved_context": response.retrieved_context,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "context_relevance": result.metadata.get("context_relevance"),
        "retrieval_latency_ms": response.metadata.get("retrieval_latency_ms"),
        "total_latency_ms": response.metadata.get("total_latency_ms"),
        "status": result.status,
    }


def run_diagnosis(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Diagnose a single evaluation dict via the existing engine."""
    from .diagnostics import diagnose_result
    from .evaluation_schema import EvaluationResult

    metrics = {
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
    }
    result = EvaluationResult(
        case_id=evaluation.get("case_id", "unknown"),
        question=evaluation.get("question", ""),
        retrieved_documents=list(evaluation.get("retrieved_documents", [])),
        retrieval_attempts=1,
        latency_ms=evaluation.get("total_latency_ms"),
        retrieval_score=evaluation.get("f1"),
        status=evaluation.get("status", "SUCCESS"),
        metadata={
            "retrieval_metrics": metrics,
            "context_relevance": evaluation.get("context_relevance"),
        },
    )
    findings = diagnose_result(result)
    return {
        "overall_status": (
            "PASS" if not findings else "ISSUES FOUND"),
        "findings": [f.to_dict() for f in findings],
    }


def compare_search_methods() -> dict[str, Any]:
    """Return the labelled Phase 4 smoke-test comparison (saved values)."""
    rows = [dict(r) for r in PHASE4_SMOKE_COMPARISON]
    best = max(rows, key=lambda r: (
        r["best_objective"] is not None, r["best_objective"] or -1))
    return {"label": PHASE4_SMOKE_LABEL, "rows": rows,
            "best_method": best["method"]}


def get_recommended_configuration() -> dict[str, Any]:
    """Recommend from measured smoke-test winners (no fabrication).

    Both random and adaptive tied at 0.7434; the shared winning
    configuration from that comparison was chunk 500 / overlap 10 /
    top_k 2. The recommendation is explicitly labelled as coming from
    the smoke-test record, not from a new optimization run.
    """
    return {
        "source": PHASE4_SMOKE_LABEL,
        "current": {"chunk_size": 300, "chunk_overlap": 50, "top_k": 5},
        "recommended": {"chunk_size": 500, "chunk_overlap": 10, "top_k": 2},
        "reason": ("Tied-highest measured objective (0.7434) with strong "
                   "retrieval F1 (0.8333) and low latency (~22 ms) in the "
                   "Phase 4 smoke-test comparison."),
    }


