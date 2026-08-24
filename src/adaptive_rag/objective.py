"""Objective function for automated retrieval configuration search.

The objective balances the *retrieval* quality dimensions that matter for a
**forced-retrieval** benchmark:

- retrieval F1 (precision/recall harmonic mean)
- context relevance
- latency (normalized)

By design, answer quality is **not** optimized in Phase 3C: retrieval runs
with generation disabled, so answer scores are not available.  This decision
is documented in ``experiments/automated_search_methodology.md``.

The objective is fully deterministic:

.. code-block:: text

    Objective = w_f1 * F1
              + w_ctx * ContextRelevance
              + w_lat * LatencyScore

Latency is normalized into ``[0, 1]`` (higher is better / faster):

.. code-block:: text

    LatencyScore = 1 / (1 + latency_ms / reference_latency_ms)

Missing metrics are handled explicitly: if a component has no value (e.g.
every case in a configuration fails), that component is excluded and the
remaining weights are renormalized so the objective stays comparable on
``[0, 1]``.  Missing values are **never fabricated**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Objective component weight keys.
RETRIEVAL_F1_WEIGHT = "retrieval_f1"
CONTEXT_RELEVANCE_WEIGHT = "context_relevance"
LATENCY_WEIGHT = "latency"

DEFAULT_WEIGHTS: dict[str, float] = {
    RETRIEVAL_F1_WEIGHT: 0.50,
    CONTEXT_RELEVANCE_WEIGHT: 0.20,
    LATENCY_WEIGHT: 0.30,
}

DEFAULT_REFERENCE_LATENCY_MS: float = 100.0


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for the objective function."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    reference_latency_ms: float = DEFAULT_REFERENCE_LATENCY_MS


@dataclass(frozen=True)
class ObjectiveResult:
    """The computed objective and its component breakdown."""

    objective_score: float | None
    retrieval_component: float | None
    context_component: float | None
    latency_component: float | None
    weights: dict[str, float]
    missing_components: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "objective_score": self.objective_score,
            "retrieval_component": self.retrieval_component,
            "context_component": self.context_component,
            "latency_component": self.latency_component,
            "weights": dict(self.weights),
            "missing_components": list(self.missing_components),
        }


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate the objective weights."""
    if not weights:
        raise ValueError("Objective weights must be provided.")
    for key, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"Objective weight {key!r} must be numeric.")
        if weight < 0:
            raise ValueError(f"Objective weight {key!r} must be non-negative.")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one objective weight must be > 0.")
    return dict(weights)


def normalize_latency(
    latency_ms: float | None,
    reference_latency_ms: float = DEFAULT_REFERENCE_LATENCY_MS,
) -> float | None:
    """Return a deterministic latency norm in ``[0, 1]``.

    ``1 / (1 + latency_ms / reference)`` maps 0 ms -> 1.0, the reference
    latency -> 0.5, and increasingly slow configurations -> 0.0.
    Returns ``None`` when latency is missing, and ``0.0`` for negative
    values (which should never occur from real measurements).
    """
    if latency_ms is None:
        return None
    if reference_latency_ms <= 0:
        raise ValueError("reference_latency_ms must be positive")
    positive = max(float(latency_ms), 0.0)
    return 1.0 / (1.0 + positive / reference_latency_ms)
def calculate_objective(
    retrieval_f1: float | None,
    context_relevance: float | None,
    latency_ms: float | None,
    weights: dict[str, float] | None = None,
    reference_latency_ms: float = DEFAULT_REFERENCE_LATENCY_MS,
) -> ObjectiveResult:
    """Compute the retrieval objective for a single configuration.

    Parameters
    ----------
    retrieval_f1:
        Averaged retrieval F1 across the configuration's cases.
    context_relevance:
        Averaged context-relevance score (or ``None``).
    latency_ms:
        Averaged latency in milliseconds (or ``None``).
    weights:
        Component weights.  Defaults to
        ``{retrieval_f1: 0.50, context_relevance: 0.20, latency: 0.30}``.
    reference_latency_ms:
        Latency reference for normalization (default 100 ms).

    Returns
    -------
    ObjectiveResult
        The objective score plus the component breakdown.  The score is
        ``None`` when no component has a value.
    """
    effective_weights = (
        _validate_weights(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    )

    latency_component = normalize_latency(latency_ms, reference_latency_ms)

    components: dict[str, float] = {}
    if retrieval_f1 is not None:
        components[RETRIEVAL_F1_WEIGHT] = retrieval_f1
    if context_relevance is not None:
        components[CONTEXT_RELEVANCE_WEIGHT] = context_relevance
    if latency_component is not None:
        components[LATENCY_WEIGHT] = latency_component

    available = [key for key in effective_weights if key in components]
    missing = [key for key in effective_weights if key not in components]

    if not available:
        return ObjectiveResult(
            objective_score=None,
            retrieval_component=retrieval_f1,
            context_component=context_relevance,
            latency_component=latency_component,
            weights=effective_weights,
            missing_components=missing,
        )

    # Renormalize over the components actually present so the weighted
    # average stays on [0, 1] regardless of which metrics are missing.
    # Absent values are never fabricated.
    total_weight = sum(effective_weights[key] for key in available)
    objective_score = sum(
        effective_weights[key] * components[key] for key in available
    ) / total_weight

    return ObjectiveResult(
        objective_score=objective_score,
        retrieval_component=retrieval_f1,
        context_component=context_relevance,
        latency_component=latency_component,
        weights=effective_weights,
        missing_components=missing,
    )