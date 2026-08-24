"""Experiment configuration for the AI Evaluation & Optimization Engine.

An :class:`ExperimentConfig` describes a set of parameters that define how
an AI system should be run during an experiment.  The configuration is
provider-independent and JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes
    ----------
    experiment_id:
        Unique identifier for the experiment.
    name:
        Human-readable name for the experiment.
    description:
        Optional description of the experiment's purpose.
    parameters:
        Dictionary of arbitrary configuration values (e.g., ``top_k``,
        ``chunk_size``, ``retrieval_strategy``).
    tags:
        Optional list of tags for categorization and filtering.
    """

    experiment_id: str
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
