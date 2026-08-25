"""Comparison between grid search and adaptive search (Phase 3D).

Given the results of a grid search run and an adaptive search run over
the SAME search space, dataset, questions, objective function and
adapter, this module produces a transparent side-by-side report.

**Search efficiency** is defined as::

    search_efficiency = best_objective_found / configurations_evaluated

**Search-space coverage** is defined as::

    coverage = configurations_evaluated / total_possible_configurations

The comparison does NOT claim one strategy is superior; it only reports
measured values.  A single tiny pilot cannot establish superiority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .adaptive_search_result import AdaptiveSearchResult
from .search_result import SearchResult


@dataclass(frozen=True)
class SearchSideSummary:
    """Measured summary of one search strategy's run."""

    strategy: str
    configurations_evaluated: int
    best_objective: float | None
    best_f1: float | None
    best_context_relevance: float | None
    fastest_latency_ms: float | None
    search_efficiency: float | None
    space_coverage: float | None
    failed_configurations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class SearchComparison:
    """Side-by-side comparison of two search strategies."""

    total_possible_configurations: int = 0
    grid: SearchSideSummary | None = None
    adaptive: SearchSideSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_possible_configurations": self.total_possible_configurations,
            "grid": self.grid.to_dict() if self.grid is not None else None,
            "adaptive": (
                self.adaptive.to_dict() if self.adaptive is not None else None
            ),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)


def summarize_search(
    strategy: str,
    evaluated: int,
    ranking: list[Any],
    total_possible: int,
) -> SearchSideSummary:
    """Build a :class:`SearchSideSummary` from a ranked entry list.

    ``ranking`` entries may be :class:`SearchRankingEntry` or
    :class:`AdaptiveRankingEntry`; both expose the same metric fields.
    Missing metrics are reported as ``None`` — never fabricated.
    """
    successful = [e for e in ranking if e.status == "SUCCESS"]

    objectives = [
        float(e.objective_score)
        for e in successful
        if e.objective_score is not None
    ]
    f1s = [
        float(e.retrieval_f1) for e in successful if e.retrieval_f1 is not None
    ]
    ctxs = [
        float(e.context_relevance)
        for e in successful
        if e.context_relevance is not None
    ]
    latencies = [
        float(e.latency_ms) for e in successful if e.latency_ms is not None
    ]
    failed = sum(1 for e in ranking if e.status != "SUCCESS")

    best_objective = max(objectives) if objectives else None
    efficiency = (
        best_objective / evaluated if evaluated > 0 and best_objective is not None else None
    )
    coverage = (
        evaluated / total_possible if total_possible > 0 else None
    )

    return SearchSideSummary(
        strategy=strategy,
        configurations_evaluated=evaluated,
        best_objective=best_objective,
        best_f1=max(f1s) if f1s else None,
        best_context_relevance=max(ctxs) if ctxs else None,
        fastest_latency_ms=min(latencies) if latencies else None,
        search_efficiency=efficiency,
        space_coverage=coverage,
        failed_configurations=failed,
    )


def compare_grid_vs_adaptive(
    grid_result: SearchResult,
    adaptive_result: AdaptiveSearchResult,
    total_possible_configurations: int | None = None,
) -> SearchComparison:
    """Compare a grid-search result with an adaptive-search result.

    ``total_possible_configurations`` defaults to the adaptive result's
    recorded space size (both searches must use the same space).
    """
    total = (
        total_possible_configurations
        if total_possible_configurations is not None
        else adaptive_result.total_possible_configurations
    )
    grid_summary = summarize_search(
        strategy="grid",
        evaluated=grid_result.completed_configurations,
        ranking=list(grid_result.ranking),
        total_possible=total,
    )
    adaptive_summary = summarize_search(
        strategy="adaptive",
        evaluated=adaptive_result.configurations_evaluated,
        ranking=list(adaptive_result.ranking),
        total_possible=total,
    )
    return SearchComparison(
        total_possible_configurations=total,
        grid=grid_summary,
        adaptive=adaptive_summary,
        metadata={
            "note": (
                "Comparison reports measured values only; it does not "
                "establish superiority of either strategy."
            ),
            "grid_search_id": grid_result.search_id,
            "adaptive_search_id": adaptive_result.search_id,
        },
    )