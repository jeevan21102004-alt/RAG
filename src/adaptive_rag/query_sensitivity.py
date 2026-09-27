"""Query-type configuration sensitivity analysis (Phase 6C).

Investigates whether SIMPLE, TECHNICAL, and MULTI_DOCUMENT queries respond
differently to retrieval parameters -- WITHOUT assuming query-aware
optimization works.

Method: one-parameter-at-a-time sweeps around a fixed centre point, using
ONLY the existing retrieval-only infrastructure:

- questions: ``data/enterprise_retrieval_questions.json``
- classification: Phase 6A ``query_classifier`` (unmodified)
- per-question evaluation: Phase 6B ``evaluate_configuration`` (unmodified,
  retrieval-only, ``enable_generation=False``)
- aggregation: Phase 6B ``aggregate_metrics`` (unmodified)
- grouping: Phase 6B ``group_by_query_type`` (unmodified)

ZERO external calls: no LLM, no Gemini, no network, no RL training.

Controlled budget (labelled, deterministic): 7 configurations x 20 questions
= 140 retrieval-only evaluations. ``run_sensitivity`` accepts smaller
``configurations`` / ``questions`` overrides so unit tests stay fast.

Baseline (centre): chunk_size=300, chunk_overlap=40, top_k=4. Every other
configuration differs from the baseline in EXACTLY ONE parameter.

All reported numbers are measured values. Interpretation is kept separate
from measurement (see ``summarize_axis`` max-min ranges).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

BASELINE_CONFIGURATION: dict[str, int] = {
    "chunk_size": 300,
    "chunk_overlap": 40,
    "top_k": 4,
}

# One-parameter sweeps (each includes the baseline for reference).
AXIS_SWEEPS: dict[str, tuple[dict[str, int], ...]] = {
    "chunk_size": (
        {"chunk_size": 100, "chunk_overlap": 40, "top_k": 4},
        {"chunk_size": 300, "chunk_overlap": 40, "top_k": 4},
        {"chunk_size": 500, "chunk_overlap": 40, "top_k": 4},
    ),
    "chunk_overlap": (
        {"chunk_size": 300, "chunk_overlap": 10, "top_k": 4},
        {"chunk_size": 300, "chunk_overlap": 40, "top_k": 4},
        {"chunk_size": 300, "chunk_overlap": 75, "top_k": 4},
    ),
    "top_k": (
        {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2},
        {"chunk_size": 300, "chunk_overlap": 40, "top_k": 4},
        {"chunk_size": 300, "chunk_overlap": 40, "top_k": 6},
    ),
}

SENSITIVITY_LABEL = (
    "Phase 6C query-type sensitivity pilot "
    "(7 isolated configs x 20 questions = 140 retrieval evaluations)"
)

METRIC_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "context_relevance",
    "retrieval_latency_ms",
    "objective_score",
)


def sensitivity_configurations() -> list[dict[str, int]]:
    """Return the 7 deterministic isolated configurations (baseline first)."""
    configs: list[dict[str, int]] = [dict(BASELINE_CONFIGURATION)]
    seen = {tuple(sorted(BASELINE_CONFIGURATION.items()))}
    for axis in ("chunk_size", "chunk_overlap", "top_k"):
        for params in AXIS_SWEEPS[axis]:
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(dict(params))
    return configs


def describe_plan() -> list[dict[str, Any]]:
    """Describe each config: which axis it isolates and what stays fixed."""
    plan: list[dict[str, Any]] = []
    for config in sensitivity_configurations():
        differing = [
            k for k in BASELINE_CONFIGURATION if config[k] != BASELINE_CONFIGURATION[k]
        ]
        if not differing:
            plan.append({
                "axis": "baseline",
                "configuration": dict(config),
                "varied": None,
                "fixed": dict(config),
            })
        else:
            axis = differing[0]
            plan.append({
                "axis": axis,
                "configuration": dict(config),
                "varied": axis,
                "fixed": {k: v for k, v in config.items() if k != axis},
            })
    return plan


def run_sensitivity(
    configurations: Sequence[Mapping[str, int]] | None = None,
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the sensitivity sweep (retrieval-only, local, deterministic)."""
    from .query_aware_evaluation import (
        aggregate_metrics,
        evaluate_configuration,
        group_by_query_type,
        load_questions,
    )

    configs = (
        sensitivity_configurations()
        if configurations is None
        else [dict(c) for c in configurations]
    )
    items = load_questions() if questions is None else [dict(q) for q in questions]
    groups = group_by_query_type(items)
    counts = {qtype: len(qitems) for qtype, qitems in groups.items()}

    config_results: list[dict[str, Any]] = []
    for config in configs:
        full = evaluate_configuration(config, items)
        by_type: dict[str, dict[str, Any]] = {}
        for qtype, qitems in groups.items():
            ids = {str(q.get("case_id", "")) for q in qitems}
            subset = [e for e in full["evaluations"] if e["case_id"] in ids]
            by_type[qtype] = aggregate_metrics(subset)
        config_results.append({
            "configuration": dict(config),
            "global": dict(full["aggregated"]),
            "by_type": by_type,
            "evaluations": full["evaluations"],
        })

    total_evals = sum(len(r["evaluations"]) for r in config_results)
    axes: dict[str, Any] = {}
    for axis, sweep in AXIS_SWEEPS.items():
        sweep_keys = {tuple(sorted(dict(s).items())) for s in sweep}
        matched = [
            r for r in config_results
            if tuple(sorted(r["configuration"].items())) in sweep_keys
        ]
        matched = sorted(matched, key=lambda r: r["configuration"][axis])
        axes[axis] = {
            "values": [r["configuration"][axis] for r in matched],
            "global": [r["global"] for r in matched],
            "by_type": {
                qtype: [r["by_type"].get(qtype) for r in matched]
                for qtype in counts
            },
        }

    return {
        "label": SENSITIVITY_LABEL,
        "budget": {
            "configurations": len(configs),
            "questions": len(items),
            "retrieval_evaluations": total_evals,
        },
        "question_counts": counts,
        "baseline": dict(BASELINE_CONFIGURATION),
        "axes": axes,
        "config_results": config_results,
    }


def summarize_axis(
    axis_result: Mapping[str, Any],
    metric: str = "objective_score",
) -> dict[str, Any]:
    """Summarize one axis with measured max-min ranges only (no claims)."""
    def _range(values: Sequence[Any]) -> dict[str, Any]:
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        if not nums:
            return {"values": list(values), "min": None, "max": None, "range": None}
        return {
            "values": list(values),
            "min": min(nums),
            "max": max(nums),
            "range": max(nums) - min(nums),
        }

    global_vals = [(a or {}).get(metric) for a in axis_result.get("global", [])]
    out: dict[str, Any] = {
        "metric": metric,
        "parameter_values": list(axis_result.get("values", [])),
        "global": _range(global_vals),
        "by_type": {},
    }
    for qtype, aggs in (axis_result.get("by_type", {}) or {}).items():
        out["by_type"][qtype] = _range([(a or {}).get(metric) for a in aggs])
    return out

