"""Search result data model for automated configuration search.

A :class:`SearchResult` captures everything a deterministic grid search
produced: the configuration list, per-configuration performance, the
objective ranking, and the best configuration.  It is JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchRankingEntry:
    """Performance summary of a single evaluated configuration."""

    experiment_id: str
    parameters: dict[str, int]
    objective_score: float | None
    retrieval_f1: float | None
    context_relevance: float | None
    latency_ms: float | None
    status: str = "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class SearchResult:
    """Aggregated result of an automated configuration search.

    Attributes
    ----------
    search_id:
        Stable identifier for this search run.
    total_configurations:
        Number of configurations in the search space.
    completed_configurations:
        Number of configurations that completed with a SUCCESS status
        (including configurations whose objective could still be computed
        from partial metrics).
    failed_configurations:
        Number of configurations that failed (no metrics could be computed
        or the experiment raised).
    ranking:
        Configurations ranked by objective score (descending), with a
        deterministic tie-break.
    best_configuration:
        The top-ranked configuration (or ``None`` if nothing completed).
    best_objective_score:
        Objective score of the best configuration.
    objective_definition:
        The objective weights and reference latency used.
    metadata:
        Search-space description, dataset, question count, adapter info,
        API error counts, and a timestamp.
    """

    search_id: str = ""
    total_configurations: int = 0
    completed_configurations: int = 0
    failed_configurations: int = 0
    ranking: list[SearchRankingEntry] = field(default_factory=list)
    best_configuration: SearchRankingEntry | None = None
    best_objective_score: float | None = None
    objective_definition: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "search_id": self.search_id,
            "total_configurations": self.total_configurations,
            "completed_configurations": self.completed_configurations,
            "failed_configurations": self.failed_configurations,
            "ranking": [entry.to_dict() for entry in self.ranking],
            "best_configuration": (
                self.best_configuration.to_dict()
                if self.best_configuration is not None
                else None
            ),
            "best_objective_score": self.best_objective_score,
            "objective_definition": dict(self.objective_definition),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)