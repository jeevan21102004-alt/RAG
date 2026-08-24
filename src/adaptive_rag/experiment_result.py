"""Experiment result data model.

An :class:`ExperimentResult` aggregates the outcomes of running an
experiment across multiple evaluation cases.  It is JSON-serializable and
does not contain any evaluation logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .experiment_config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentResult:
    """Aggregated result of running an experiment.

    Attributes
    ----------
    experiment_id:
        Identifier matching the :class:`ExperimentConfig`.
    config:
        The configuration that was used for this experiment.
    total_cases:
        Total number of evaluation cases run.
    successful_cases:
        Number of cases that completed without error.
    failed_cases:
        Number of cases that failed (non-SUCCESS status).
    average_answer_score:
        Mean answer_score across all cases (or ``None``).
    average_retrieval_score:
        Mean retrieval_score across all cases (or ``None``).
    average_decision_score:
        Mean decision_score across all cases (or ``None``).
    average_overall_score:
        Mean overall_score across all cases (or ``None``).
    average_latency_ms:
        Mean latency in milliseconds across all cases (or ``None``).
    total_retrieval_attempts:
        Sum of retrieval_attempts across all cases.
    diagnostics:
        Aggregated system diagnosis from the diagnostic engine.
    metadata:
        Additional arbitrary metadata.
    """

    experiment_id: str
    config: ExperimentConfig
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    average_answer_score: float | None = None
    average_retrieval_score: float | None = None
    average_decision_score: float | None = None
    average_overall_score: float | None = None
    average_latency_ms: float | None = None
    total_retrieval_attempts: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
