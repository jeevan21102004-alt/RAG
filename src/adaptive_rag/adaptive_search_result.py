"""Adaptive search result data model for Phase 3D.

An :class:`AdaptiveSearchResult` captures everything a deterministic
adaptive configuration search produced: the evaluation order, the
selection reasons, per-configuration performance, and the best
configuration found.  It is JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdaptiveRankingEntry:
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


@dataclass(frozen=True)
class EvaluationOrderEntry:
    """Record of one configuration evaluated by the adaptive engine.

    Attributes
    ----------
    experiment_id:
        Deterministic experiment identifier.
    parameters:
        The parameter values that were evaluated.
    objective_score:
        Objective score after evaluation (``None`` when the run failed).
    retrieval_f1, context_relevance, latency_ms:
        Component metrics (``None`` when missing).
    status:
        ``"SUCCESS"`` or a ``CONFIG_ERROR: ...`` description.
    selection_reason:
        Transparent, human-readable reason why this configuration was
        selected at this point in the search.
    """

    experiment_id: str
    parameters: dict[str, int]
    objective_score: float | None
    retrieval_f1: float | None
    context_relevance: float | None
    latency_ms: float | None
    status: str
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class AdaptiveSearchResult:
    """Aggregated result of an adaptive configuration search.

    Attributes
    ----------
    search_id:
        Stable identifier for this search run.
    search_space:
        Serialized description of the search space used.
    total_possible_configurations:
        Size of the full search space (e.g. 175).
    configurations_evaluated:
        Number of configurations actually evaluated.
    configurations_remaining:
        ``total_possible_configurations - configurations_evaluated``.
    failed_configurations:
        Number of evaluated configurations that failed.
    best_configuration:
        Top-ranked evaluated configuration (or ``None``).
    best_objective_score:
        Objective score of the best configuration.
    ranking:
        Evaluated configurations ranked by objective (descending), with a
        deterministic tie-break by experiment_id.
    evaluation_order:
        The exact order in which configurations were selected and
        evaluated, including the selection reason for each.
    selection_reasons:
        Mapping of experiment_id -> selection reason (redundant with
        ``evaluation_order`` but convenient for lookup).
    objective_definition:
        The objective weights and reference latency used.
    metadata:
        Exploration bonus, dataset, question count, API errors, etc.
    """

    search_id: str = ""
    search_space: dict[str, Any] = field(default_factory=dict)
    total_possible_configurations: int = 0
    configurations_evaluated: int = 0
    configurations_remaining: int = 0
    failed_configurations: int = 0
    best_configuration: AdaptiveRankingEntry | None = None
    best_objective_score: float | None = None
    ranking: list[AdaptiveRankingEntry] = field(default_factory=list)
    evaluation_order: list[EvaluationOrderEntry] = field(default_factory=list)
    selection_reasons: dict[str, str] = field(default_factory=dict)
    objective_definition: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "search_id": self.search_id,
            "search_space": dict(self.search_space),
            "total_possible_configurations": self.total_possible_configurations,
            "configurations_evaluated": self.configurations_evaluated,
            "configurations_remaining": self.configurations_remaining,
            "failed_configurations": self.failed_configurations,
            "best_configuration": (
                self.best_configuration.to_dict()
                if self.best_configuration is not None
                else None
            ),
            "best_objective_score": self.best_objective_score,
            "ranking": [entry.to_dict() for entry in self.ranking],
            "evaluation_order": [entry.to_dict() for entry in self.evaluation_order],
            "selection_reasons": dict(self.selection_reasons),
            "objective_definition": dict(self.objective_definition),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)