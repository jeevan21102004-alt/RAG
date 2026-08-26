"""Numerical RL state for retrieval configuration optimization (Phase 4).

The state summarizes the optimization history as a **fixed-size vector of
15 numerical features** (no LLM, no raw text):

.. code-block:: text

    idx  feature                                   range
    ---  ---------------------------------------   -----
    0    current best objective                    [0, 1]
    1    current best F1                           [0, 1]
    2    current best context relevance            [0, 1]
    3    current best latency score                [0, 1]
    4    fraction of budget consumed               [0, 1]
    5    fraction of search space evaluated        [0, 1]
    6    successful evaluations / budget           [0, 1]
    7    failed evaluations / budget               [0, 1]
    8    average objective of evaluated configs    [0, 1]
    9    best-objective improvement (last step)    [0, 1]
    10   fraction of space still unexplored        [0, 1]
    11   last action improved the best (0/1)       {0, 1}
    12   previous action chunk_size (index-norm)   [0, 1]
    13   previous action chunk_overlap (idx-norm)  [0, 1]
    14   previous action top_k (index-normalized)  [0, 1]

``OPTIMIZATION_STATE_DIM = 15`` is fixed and documented.  Missing metrics
are encoded as ``0.0`` — never fabricated.  Index normalization maps a
parameter value to ``(index in sorted value list) / (len - 1)`` so every
parameter contributes a comparable value in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPTIMIZATION_STATE_DIM: int = 15

# Parameter names in the canonical order used for features 12-14.
_STATE_PARAMETERS: tuple[str, ...] = ("chunk_size", "chunk_overlap", "top_k")


def _normalize_index(value: int | None, values: list[int]) -> float:
    """Index-normalize *value* within the sorted list *values* to [0, 1]."""
    if value is None or len(values) <= 1:
        return 0.0
    try:
        index = values.index(value)
    except ValueError:
        return 0.0
    return index / (len(values) - 1)


@dataclass
class OptimizationState:
    """Snapshot of optimization history used as the RL state.

    All numeric fields are already normalized to ``[0, 1]`` so the vector
    is directly consumable by the policy network.
    """

    best_objective: float = 0.0
    best_f1: float = 0.0
    best_context_relevance: float = 0.0
    best_latency_score: float = 0.0
    budget_consumed_fraction: float = 0.0
    space_evaluated_fraction: float = 0.0
    successful_fraction: float = 0.0
    failed_fraction: float = 0.0
    average_objective: float = 0.0
    last_improvement: float = 0.0
    unexplored_fraction: float = 1.0
    last_action_improved: float = 0.0
    prev_chunk_size: float = 0.0
    prev_chunk_overlap: float = 0.0
    prev_top_k: float = 0.0

    def to_vector(self) -> list[float]:
        """Return the fixed-size state vector (see module docstring)."""
        return [
            self.best_objective,
            self.best_f1,
            self.best_context_relevance,
            self.best_latency_score,
            self.budget_consumed_fraction,
            self.space_evaluated_fraction,
            self.successful_fraction,
            self.failed_fraction,
            self.average_objective,
            self.last_improvement,
            self.unexplored_fraction,
            self.last_action_improved,
            self.prev_chunk_size,
            self.prev_chunk_overlap,
            self.prev_top_k,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "dimension": OPTIMIZATION_STATE_DIM,
            "vector": self.to_vector(),
            "features": {
                "best_objective": self.best_objective,
                "best_f1": self.best_f1,
                "best_context_relevance": self.best_context_relevance,
                "best_latency_score": self.best_latency_score,
                "budget_consumed_fraction": self.budget_consumed_fraction,
                "space_evaluated_fraction": self.space_evaluated_fraction,
                "successful_fraction": self.successful_fraction,
                "failed_fraction": self.failed_fraction,
                "average_objective": self.average_objective,
                "last_improvement": self.last_improvement,
                "unexplored_fraction": self.unexplored_fraction,
                "last_action_improved": self.last_action_improved,
                "prev_chunk_size": self.prev_chunk_size,
                "prev_chunk_overlap": self.prev_chunk_overlap,
                "prev_top_k": self.prev_top_k,
            },
        }


def optimization_state_to_vector(
    state: OptimizationState,
) -> list[float]:
    """Return the fixed-size numeric vector for *state*.

    The vector length is always :data:`OPTIMIZATION_STATE_DIM` (15).
    """
    vector = state.to_vector()
    if len(vector) != OPTIMIZATION_STATE_DIM:
        raise ValueError(
            f"State vector has {len(vector)} features; "
            f"expected {OPTIMIZATION_STATE_DIM}."
        )
    return vector