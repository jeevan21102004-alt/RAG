"""Deterministic configuration generator for grid search.

This module converts a :class:`SearchSpace` into a flat list of
:class:`ExperimentConfig` objects.  Generation is fully deterministic: given
the same search space, it always produces the same configurations in the
same order, with stable, unique experiment IDs.

No randomness is used.
"""

from __future__ import annotations

from typing import Any

from .experiment_config import ExperimentConfig
from .search_space import SearchSpace


def generate_configurations(
    search_space: SearchSpace,
    prefix: str = "grid",
) -> list[ExperimentConfig]:
    """Generate every :class:`ExperimentConfig` in the search space.

    Parameters
    ----------
    search_space:
        The search space to expand.
    prefix:
        String prefix used to build deterministic experiment IDs.

    Returns
    -------
    list[ExperimentConfig]
        One configuration per parameter combination, in deterministic
        order, with unique IDs.
    """
    search_space.raise_if_invalid()
    combos = list(search_space.generate_parameter_combinations())

    configs: list[ExperimentConfig] = []
    width = len(str(len(combos)))
    for index, parameters in enumerate(combos, start=1):
        experiment_id = f"{prefix}-{index:0{width}d}"
        name = _build_name(parameters, index)
        description = _build_description(parameters)
        configs.append(
            ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                parameters=dict(parameters),
                tags=["grid_search", search_space.name],
            )
        )
    return configs


def _build_name(parameters: dict[str, Any], index: int) -> str:
    """Build a compact human-readable name for a configuration."""
    parts = ", ".join(f"{key}={value}" for key, value in parameters.items())
    return f"Config {index} ({parts})"


def _build_description(parameters: dict[str, Any]) -> str:
    """Build a descriptive sentence for a configuration."""
    chunks = "; ".join(f"{key}={value}" for key, value in parameters.items())
    return f"Automated grid-search configuration with {chunks}."