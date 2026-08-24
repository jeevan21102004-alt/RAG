"""Experiment comparison utilities.

This module provides functions to compare multiple
:class:`ExperimentResult` objects and rank them.

This is **comparison only** — no optimization or parameter search is
performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .experiment_result import ExperimentResult


@dataclass(frozen=True)
class ExperimentComparison:
    """Structured comparison of multiple experiment results.

    Attributes
    ----------
    experiments:
        List of experiment IDs included in the comparison.
    best_by_overall_score:
        ID of the experiment with the highest average overall score.
    best_by_answer_score:
        ID of the experiment with the highest average answer score.
    best_by_retrieval_score:
        ID of the experiment with the highest average retrieval score.
    fastest_experiment:
        ID of the experiment with the lowest average latency.
    ranking:
        List of experiment IDs ranked by average overall score
        (descending).
    """

    experiments: list[str] = field(default_factory=list)
    best_by_overall_score: str | None = None
    best_by_answer_score: str | None = None
    best_by_retrieval_score: str | None = None
    fastest_experiment: str | None = None
    ranking: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_max(
    results: list[ExperimentResult],
    key: str,
) -> str | None:
    """Return the experiment_id with the highest value for *key*.

    Returns ``None`` if no results have a non-None value for *key*.
    """
    candidates = [
        (r.experiment_id, getattr(r, key))
        for r in results
        if getattr(r, key) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def _safe_min(
    results: list[ExperimentResult],
    key: str,
) -> str | None:
    """Return the experiment_id with the lowest value for *key*.

    Returns ``None`` if no results have a non-None value for *key*.
    """
    candidates = [
        (r.experiment_id, getattr(r, key))
        for r in results
        if getattr(r, key) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def compare_experiments(
    results: list[ExperimentResult],
) -> ExperimentComparison:
    """Compare multiple experiment results and rank them.

    Ranking is primarily based on ``average_overall_score`` (descending).
    Experiments with a ``None`` overall score are ranked last.

    Parameters
    ----------
    results:
        List of :class:`ExperimentResult` objects to compare.

    Returns
    -------
    ExperimentComparison
        Structured comparison with best experiments and ranking.
    """
    experiment_ids = [r.experiment_id for r in results]

    # Ranking by overall score (descending), None values last
    ranked = sorted(
        results,
        key=lambda r: (
            r.average_overall_score is not None,
            r.average_overall_score if r.average_overall_score is not None else -1,
        ),
        reverse=True,
    )
    ranking = [r.experiment_id for r in ranked]

    return ExperimentComparison(
        experiments=experiment_ids,
        best_by_overall_score=_safe_max(results, "average_overall_score"),
        best_by_answer_score=_safe_max(results, "average_answer_score"),
        best_by_retrieval_score=_safe_max(results, "average_retrieval_score"),
        fastest_experiment=_safe_min(results, "average_latency_ms"),
        ranking=ranking,
    )
