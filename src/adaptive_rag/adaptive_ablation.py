"""Adaptive retrieval ablation (Phase 6E pilot).

Compares FIXED_2 / FIXED_4 / FIXED_6 against the existing Phase 6D
``AdaptiveDepthPolicy`` on the 20-question enterprise benchmark, using ONLY
the existing retrieval-only infrastructure (unmodified):

- ``evaluate_configuration`` / ``aggregate_metrics`` (Phase 6B)
- ``AdaptiveDepthPolicy`` / ``decide_for_question`` (Phase 6D)
- ``RetrievalBenchmarkAdapter(enable_generation=False)``

Fixed context: chunk_size=300, chunk_overlap=40 (the 6C centre).

Budget (labelled, deterministic): 4 distinct top_k values {2,3,4,6} x 20
questions = 80 retrieval-only evaluations. The adaptive arm reuses the
cached per-question rows (no duplicate retrieval calls).

ZERO external calls: no LLM, no Gemini, no network, no RL training.

All numbers reported are measured values; interpretation is separate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

POLICY_NAMES: tuple[str, ...] = ("FIXED_2", "FIXED_4", "FIXED_6", "ADAPTIVE")

FIXED_TOP_KS: dict[str, int] = {
    "FIXED_2": 2,
    "FIXED_4": 4,
    "FIXED_6": 6,
}

ABLATION_LABEL = (
    "Phase 6E adaptive-depth ablation "
    "(FIXED_2 vs FIXED_4 vs FIXED_6 vs ADAPTIVE, 80 retrieval evaluations)"
)

METRIC_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "context_relevance",
    "retrieval_latency_ms",
    "objective_score",
)


def describe_policies(policy: Any = None) -> dict[str, dict[str, Any]]:
    """Describe the four policies (fixed + adaptive mapping)."""
    from .adaptive_retrieval import AdaptiveDepthPolicy

    active = policy or AdaptiveDepthPolicy()
    out: dict[str, dict[str, Any]] = {}
    for name, top_k in FIXED_TOP_KS.items():
        out[name] = {
            "kind": "fixed",
            "top_k": top_k,
            "configuration": {
                "chunk_size": active.chunk_size,
                "chunk_overlap": active.chunk_overlap,
                "top_k": top_k,
            },
        }
    out["ADAPTIVE"] = {
        "kind": "adaptive",
        "mapping": {
            "LOW": active.low_top_k,
            "MEDIUM": active.medium_top_k,
            "HIGH": active.high_top_k,
        },
        "context": {
            "chunk_size": active.chunk_size,
            "chunk_overlap": active.chunk_overlap,
        },
    }
    return out


def top_k_for_policy(
    policy_name: str,
    decision: Mapping[str, Any] | None = None,
    policy: Any = None,
) -> int:
    """Return the top_k a policy assigns (fixed ignores the decision)."""
    if policy_name in FIXED_TOP_KS:
        return FIXED_TOP_KS[policy_name]
    if policy_name == "ADAPTIVE":
        if decision is None:
            raise ValueError("ADAPTIVE requires a per-question decision")
        return int(decision["top_k"])
    raise ValueError(f"Unknown policy: {policy_name!r}")


def run_ablation(
    policy: Any = None,
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all four policies (retrieval-only, local, deterministic)."""
    from .adaptive_retrieval import AdaptiveDepthPolicy, decide_for_question
    from .query_aware_evaluation import (
        aggregate_metrics,
        evaluate_configuration,
        group_by_query_type,
        load_questions,
    )

    active = policy or AdaptiveDepthPolicy()
    items = load_questions() if questions is None else [dict(q) for q in questions]
    decisions = [decide_for_question(item, active) for item in items]
    groups = group_by_query_type(items)
    needed = sorted({*FIXED_TOP_KS.values(),
                     *[int(d["top_k"]) for d in decisions]})
    cache: dict[int, dict[str, Any]] = {}
    for top_k in needed:
        cache[top_k] = evaluate_configuration(
            {"chunk_size": active.chunk_size,
             "chunk_overlap": active.chunk_overlap, "top_k": top_k},
            items,
        )
    policies: dict[str, Any] = {}
    for name in POLICY_NAMES:
        if name == "ADAPTIVE":
            evals = []
            for decision in decisions:
                match = next(
                    e for e in cache[int(decision["top_k"])]["evaluations"]
                    if e["case_id"] == decision["case_id"])
                evals.append(match)
        else:
            evals = cache[FIXED_TOP_KS[name]]["evaluations"]
        by_type = {}
        for qtype, qitems in groups.items():
            ids = {str(q.get("case_id", "")) for q in qitems}
            by_type[qtype] = aggregate_metrics(
                [e for e in evals if e["case_id"] in ids])
        policies[name] = {"aggregated": aggregate_metrics(evals),
                          "by_type": by_type, "evaluations": evals}
    fixed2 = policies["FIXED_2"]["aggregated"]
    differences = {}
    for name in ("FIXED_2", "FIXED_4", "FIXED_6"):
        base = policies[name]["aggregated"]
        differences[f"ADAPTIVE_minus_{name}"] = {
            key: (policies["ADAPTIVE"]["aggregated"].get(key)
                  - base.get(key)
                  if isinstance(
                      policies["ADAPTIVE"]["aggregated"].get(key), (int, float))
                  and isinstance(base.get(key), (int, float)) else None)
            for key in METRIC_KEYS}
    total_evals = sum(len(cache[k]["evaluations"]) for k in needed)
    top_k_counts: dict[int, int] = {}
    for decision in decisions:
        top_k_counts[int(decision["top_k"])] = (
            top_k_counts.get(int(decision["top_k"]), 0) + 1)
    return {
        "label": ABLATION_LABEL,
        "policies": describe_policies(active),
        "budget": {"distinct_top_k_values": needed,
                   "questions": len(items),
                   "retrieval_evaluations": total_evals},
        "question_counts": {q: len(v) for q, v in groups.items()},
        "adaptive_top_k_counts": top_k_counts,
        "results": policies,
        "differences": differences,
        "fixed2_reference": {k: fixed2.get(k) for k in METRIC_KEYS},
    }

