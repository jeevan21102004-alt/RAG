"""JSON-based storage for experiment results.

This module provides simple save/load utilities for
:class:`ExperimentResult` objects.  Results are stored as JSON files in
the ``experiments/results/`` directory.

No database is used.  No secrets are stored.
"""

from __future__ import annotations

import json
from pathlib import Path

from .experiment_result import ExperimentResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def save_experiment_result(
    result: ExperimentResult,
    path: Path | str | None = None,
) -> Path:
    """Save an :class:`ExperimentResult` as a JSON file.

    Parameters
    ----------
    result:
        The experiment result to save.
    path:
        Optional file path.  If not provided, the result is saved to
        ``experiments/results/{experiment_id}.json``.

    Returns
    -------
    Path
        The path where the result was saved.
    """
    results_path = Path(path) if path else RESULTS_DIR / f"{result.experiment_id}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(result.to_json(), encoding="utf-8")
    return results_path


def load_experiment_result(path: Path | str) -> ExperimentResult:
    """Load an :class:`ExperimentResult` from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    ExperimentResult
        The deserialized experiment result.
    """
    from .experiment_config import ExperimentConfig

    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))

    config_data = data["config"]
    config = ExperimentConfig(
        experiment_id=config_data["experiment_id"],
        name=config_data["name"],
        description=config_data.get("description", ""),
        parameters=config_data.get("parameters", {}),
        tags=config_data.get("tags", []),
    )

    return ExperimentResult(
        experiment_id=data["experiment_id"],
        config=config,
        total_cases=data.get("total_cases", 0),
        successful_cases=data.get("successful_cases", 0),
        failed_cases=data.get("failed_cases", 0),
        average_answer_score=data.get("average_answer_score"),
        average_retrieval_score=data.get("average_retrieval_score"),
        average_decision_score=data.get("average_decision_score"),
        average_overall_score=data.get("average_overall_score"),
        average_latency_ms=data.get("average_latency_ms"),
        total_retrieval_attempts=data.get("total_retrieval_attempts", 0),
        diagnostics=data.get("diagnostics", {}),
        metadata=data.get("metadata", {}),
    )
