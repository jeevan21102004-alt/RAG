"""Deterministic adaptive retrieval depth (Phase 6D pilot).

Selects ``top_k`` per query from a deterministic difficulty estimate built
ONLY on the existing Phase 6A ``QueryFeatures`` -- no LLM, no Gemini, no
network, no RL training.

Difficulty rules (explicit, applied in order):

1. HIGH -- the query needs multiple documents
   (``num_relevant_documents > 1`` or ``has_multi_suffix``), or it spans
   two or more enterprise domains (``num_domains_mentioned >= 2``).
2. MEDIUM -- the query carries technical vocabulary
   (``technical_term_hits >= 1``) or an explicit conjunction marker
   (``conjunction_count >= 1``), and rule 1 did not fire.
3. LOW -- everything else.

Difficulty -> top_k mapping (configurable ``AdaptiveDepthPolicy``).

Phase 6C evidence constrains the defaults: in the isolated top_k sweep
(300/40/{2,4,6}) top_k=2 had the best F1 and top_k=6 the worst, with the
same direction in SIMPLE, TECHNICAL, and MULTI_DOCUMENT. The mechanism
therefore does NOT equate "harder = retrieve more". Defaults:

- LOW -> top_k=2 (best measured F1 in the sweep)
- MEDIUM -> top_k=3 (one step up, inside the tested range)
- HIGH -> top_k=4 (centre of the tested range, NOT the untested top_k=6)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class RetrievalDifficulty(str, Enum):
    """Deterministic difficulty tiers for adaptive retrieval depth."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class AdaptiveDepthPolicy:
    """Configurable difficulty -> top_k mapping plus fixed sweep context."""

    chunk_size: int = 300
    chunk_overlap: int = 40
    low_top_k: int = 2
    medium_top_k: int = 3
    high_top_k: int = 4
    baseline_top_k: int = 4

    def top_k_for(self, difficulty: RetrievalDifficulty) -> int:
        """Return the configured top_k for a difficulty tier."""
        if difficulty is RetrievalDifficulty.LOW:
            return self.low_top_k
        if difficulty is RetrievalDifficulty.MEDIUM:
            return self.medium_top_k
        if difficulty is RetrievalDifficulty.HIGH:
            return self.high_top_k
        raise ValueError(f"Unknown difficulty: {difficulty!r}")

    def configuration_for(self, difficulty: RetrievalDifficulty) -> dict[str, int]:
        """Return the full retrieval config for a difficulty tier."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k_for(difficulty),
        }

    def baseline_configuration(self) -> dict[str, int]:
        """Return the fixed top_k baseline configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.baseline_top_k,
        }


def estimate_difficulty_from_features(features: Any) -> RetrievalDifficulty:
    """Estimate difficulty deterministically from Phase 6A features."""
    num_docs = int(getattr(features, "num_relevant_documents", 0) or 0)
    multi_suffix = bool(getattr(features, "has_multi_suffix", False))
    num_domains = int(getattr(features, "num_domains_mentioned", 0) or 0)
    tech_hits = int(getattr(features, "technical_term_hits", 0) or 0)
    conjunctions = int(getattr(features, "conjunction_count", 0) or 0)
    if num_docs > 1 or multi_suffix or num_domains >= 2:
        return RetrievalDifficulty.HIGH
    if tech_hits >= 1 or conjunctions >= 1:
        return RetrievalDifficulty.MEDIUM
    return RetrievalDifficulty.LOW


def estimate_difficulty(
    question: Any = "",
    category: Any = "",
    relevant_documents: Any = None,
) -> RetrievalDifficulty:
    """Estimate difficulty directly from question inputs (features reused)."""
    from .query_classifier import extract_query_features

    return estimate_difficulty_from_features(
        extract_query_features(question, category, relevant_documents)
    )


def select_top_k(
    question: Any = "",
    category: Any = "",
    relevant_documents: Any = None,
    policy: AdaptiveDepthPolicy | None = None,
) -> int:
    """Select top_k for one question via difficulty (deterministic)."""
    active = policy or AdaptiveDepthPolicy()
    _difficulty = estimate_difficulty(question, category, relevant_documents)
    return active.top_k_for(_difficulty)


def explain_decision(
    q: Any = "",
    category: Any = "",
    relevant_documents: Any = None,
    policy: AdaptiveDepthPolicy | None = None,
) -> dict[str, Any]:
    """Explain the difficulty/top_k decision with responsible rules."""
    from .query_classifier import extract_query_features

    active = policy or AdaptiveDepthPolicy()
    features = extract_query_features(q, category, relevant_documents)
    difficulty = estimate_difficulty_from_features(features)
    reasons: list[str] = []
    if features.num_relevant_documents > 1 or features.has_multi_suffix:
        reasons.append("multi-document evidence")
    if features.num_domains_mentioned >= 2:
        reasons.append("cross-domain evidence")
    if difficulty is RetrievalDifficulty.MEDIUM:
        if features.technical_term_hits >= 1:
            reasons.append("technical vocabulary")
        if features.conjunction_count >= 1:
            reasons.append("conjunction marker present")
    if difficulty is RetrievalDifficulty.LOW:
        reasons.append("no difficulty markers")
    cfg = active.configuration_for(difficulty)
    feats = {
        "num_relevant_documents": features.num_relevant_documents,
        "has_multi_suffix": features.has_multi_suffix,
        "num_domains_mentioned": features.num_domains_mentioned,
        "technical_term_hits": features.technical_term_hits,
        "conjunction_count": features.conjunction_count,
    }
    return {
        "difficulty": difficulty.value,
        "top_k": active.top_k_for(difficulty),
        "configuration": cfg,
        "rules_fired": reasons,
        "features": feats,
    }


def decide_for_question(
    item: Mapping[str, Any],
    policy: AdaptiveDepthPolicy | None = None,
) -> dict[str, Any]:
    """Decide difficulty/top_k for one benchmark row dict."""
    explanation = explain_decision(
        item.get("question", ""),
        item.get("category", ""),
        item.get("relevant_documents", []),
        policy,
    )
    return {"case_id": str(item.get("case_id", "")), **explanation}


def evaluate_policies(
    policy: AdaptiveDepthPolicy | None = None,
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare fixed baseline vs adaptive policy (retrieval-only pilot)."""
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
    needed = sorted({active.baseline_top_k}
                    | {int(d["top_k"]) for d in decisions})
    cache: dict[int, dict[str, Any]] = {}
    for top_k in needed:
        cache[top_k] = evaluate_configuration(
            {"chunk_size": active.chunk_size,
             "chunk_overlap": active.chunk_overlap, "top_k": top_k},
            items,
        )
    baseline = cache[active.baseline_top_k]
    adaptive_evals: list[dict[str, Any]] = []
    for decision in decisions:
        match = next(
            e for e in cache[int(decision["top_k"])]["evaluations"]
            if e["case_id"] == decision["case_id"]
        )
        adaptive_evals.append(match)
    adaptive_agg = aggregate_metrics(adaptive_evals)
    fixed = dict(baseline["aggregated"])
    keys = ("precision", "recall", "f1", "context_relevance",
            "retrieval_latency_ms", "objective_score")
    delta = {
        key: (adaptive_agg.get(key) - fixed.get(key)
              if isinstance(adaptive_agg.get(key), (int, float))
              and isinstance(fixed.get(key), (int, float)) else None)
        for key in keys
    }
    difficulty_counts: dict[str, int] = {}
    top_k_counts: dict[int, int] = {}
    for decision in decisions:
        difficulty_counts[decision["difficulty"]] = (
            difficulty_counts.get(decision["difficulty"], 0) + 1)
        top_k_counts[int(decision["top_k"])] = (
            top_k_counts.get(int(decision["top_k"]), 0) + 1)
    by_type: dict[str, dict[str, Any]] = {}
    adaptive_by_type: dict[str, dict[str, Any]] = {}
    for qtype, qitems in groups.items():
        ids = {str(q.get("case_id", "")) for q in qitems}
        by_type[qtype] = aggregate_metrics(
            [e for e in baseline["evaluations"] if e["case_id"] in ids])
        adaptive_by_type[qtype] = aggregate_metrics(
            [e for e in adaptive_evals if e["case_id"] in ids])
    total_evals = sum(len(cache[k]["evaluations"]) for k in needed)
    return {
        "label": "Phase 6D adaptive-depth pilot (fixed vs adaptive top_k)",
        "policy": {
            "chunk_size": active.chunk_size,
            "chunk_overlap": active.chunk_overlap,
            "mapping": {"LOW": active.low_top_k,
                        "MEDIUM": active.medium_top_k,
                        "HIGH": active.high_top_k},
            "baseline_top_k": active.baseline_top_k,
        },
        "budget": {"distinct_top_k_values": needed,
                   "questions": len(items),
                   "retrieval_evaluations": total_evals},
        "question_counts": {q: len(v) for q, v in groups.items()},
        "difficulty_counts": difficulty_counts,
        "top_k_counts": top_k_counts,
        "decisions": decisions,
        "fixed": {"configuration": active.baseline_configuration(),
                  "aggregated": fixed, "by_type": by_type},
        "adaptive": {"aggregated": adaptive_agg,
                     "by_type": adaptive_by_type},
        "delta_adaptive_minus_fixed": delta,
    }

