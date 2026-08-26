"""Common result format for configuration optimization methods (Phase 4).

Every method (random, grid, adaptive, RL) produces an
:class:`OptimizationResult`.  The same fields are populated for each
method so the four are compared fairly on identical metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OptimizationEntry:
    """One evaluated configuration within an optimization run."""

    method: str
    step: int
    action_id: int | None
    parameters: dict[str, int]
    objective_score: float | None
    retrieval_f1: float | None
    context_relevance: float | None
    latency_ms: float | None
    reward: float | None = None
    status: str = "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Aggregated result for one optimization method."""

    method: str = ""
    seed: int | None = None
    budget: int = 0
    search_space: dict[str, Any] = field(default_factory=dict)
    training_question_ids: list[str] = field(default_factory=list)
    test_question_ids: list[str] = field(default_factory=list)
    configurations_evaluated: int = 0
    evaluation_order: list[OptimizationEntry] = field(default_factory=list)
    best_configuration: OptimizationEntry | None = None
    best_objective: float | None = None
    best_f1: float | None = None
    best_context_relevance: float | None = None
    best_latency_ms: float | None = None
    cumulative_reward: float | None = None
    final_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def comparison_metrics(self) -> dict[str, Any]:
        """Return the standard comparison metrics for this method."""
        return {
            "method": self.method,
            "budget": self.budget,
            "configurations_evaluated": self.configurations_evaluated,
            "best_objective": self.best_objective,
            "best_f1": self.best_f1,
            "best_context_relevance": self.best_context_relevance,
            "best_latency_ms": self.best_latency_ms,
            "cumulative_reward": self.cumulative_reward,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "budget": self.budget,
            "search_space": dict(self.search_space),
            "training_question_ids": list(self.training_question_ids),
            "test_question_ids": list(self.test_question_ids),
            "configurations_evaluated": self.configurations_evaluated,
            "evaluation_order": [
                entry.to_dict() for entry in self.evaluation_order
            ],
            "best_configuration": (
                self.best_configuration.to_dict()
                if self.best_configuration is not None
                else None
            ),
            "best_objective": self.best_objective,
            "best_f1": self.best_f1,
            "best_context_relevance": self.best_context_relevance,
            "best_latency_ms": self.best_latency_ms,
            "cumulative_reward": self.cumulative_reward,
            "final_reward": self.final_reward,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)