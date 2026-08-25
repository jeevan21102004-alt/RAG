"""Deterministic parameter search space for AdaptiveRAG retrieval.

This module defines the :class:`SearchSpace`, a structured, configurable
representation of the parameter values that automated configuration search
is allowed to probe.  Phase 3C performs **deterministic grid search** only —
the values below define the grid, and the search algorithm never hardcodes
them.

The search space is expressed as a mapping from parameter name to an
ordered list of candidate values:

- ``chunk_size``: number of words per chunk.
- ``chunk_overlap``: number of words shared between adjacent chunks.
- ``top_k``: number of chunks retrieved.

Validity is enforced: every parameter value must be a positive integer,
lists must be non-empty, duplicate values are rejected, and
``chunk_overlap`` must be strictly smaller than ``chunk_size`` for every
combination (a chunk cannot overlap itself entirely).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterator

# The canonical AdaptiveRAG parameter names that the search engine knows
# how to pass into an ExperimentConfig.
SUPPORTED_PARAMETERS: tuple[str, ...] = (
    "chunk_size",
    "chunk_overlap",
    "top_k",
)

# Size of chunk_overlap must be less than chunk_size.  This constraint is
# expressed as a pair of parameter names.
_OVERLAP_LESS_THAN = ("chunk_size", "chunk_overlap")


@dataclass
class SearchSpace:
    """A deterministic, configurable search space.

    Attributes
    ----------
    name:
        Human-readable identifier for the search space (used to derive
        experiment IDs).
    parameter_values:
        Ordered mapping of parameter name -> candidate values.  The
        iteration order is preserved so that grid generation is fully
        deterministic.
    """

    name: str = "search"
    parameter_values: dict[str, list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []

        for key in self.parameter_values:
            values = self.parameter_values[key]
            if not isinstance(values, list) or not values:
                errors.append(f"{key}: must be a non-empty list of integers")
                continue
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                errors.append(f"{key}: all values must be integers")
                continue
            if not all(v > 0 for v in values):
                errors.append(f"{key}: all values must be positive (got {values})")
                continue
            if len(set(values)) != len(values):
                errors.append(f"{key}: values must be unique (got {values})")
                continue
            # Keep deterministic: stable ordering regardless of input order.
            if values != sorted(values):
                errors.append(f"{key}: values must be sorted ascending (got {values})")

        size_key, overlap_key = _OVERLAP_LESS_THAN
        chunk_sizes = self.parameter_values.get(size_key, [])
        overlaps = self.parameter_values.get(overlap_key, [])
        for overlap in overlaps:
            for chunk_size in chunk_sizes:
                try:
                    if int(overlap) >= int(chunk_size):
                        errors.append(
                            f"invalid configuration: chunk_overlap={overlap} "
                            f">= chunk_size={chunk_size}"
                        )
                except (ValueError, TypeError):
                    pass  # already reported as non-integer above

        return errors

    def is_valid(self) -> bool:
        """Return ``True`` if the search space is valid."""
        return not self.validate()

    def raise_if_invalid(self) -> None:
        """Raise :class:`ValueError` with every error if the space is invalid."""
        errors = self.validate()
        if errors:
            raise ValueError("Invalid search space:\n  - " + "\n  - ".join(errors))
# ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_parameter_combinations(self) -> Iterator[dict[str, int]]:
        """Yield every parameter combination in deterministic order.

        Combinations are produced in the exact order of the parameter lists
        using :func:`itertools.product`, so the same search space always
        yields the same sequence of configurations.
        """
        self.raise_if_invalid()
        keys = list(self.parameter_values)
        lists = [self.parameter_values[key] for key in keys]
        for values in product(*lists):
            yield dict(zip(keys, values))

    def count_combinations(self) -> int:
        """Return the total number of configurations in this space."""
        total = 1
        for values in self.parameter_values.values():
            if not values:
                return 0
            total *= len(values)
        return total

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "parameter_values": {
                key: list(values) for key, values in self.parameter_values.items()
            },
        }

    # ------------------------------------------------------------------
    # Pre-built spaces
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SearchSpace":
        """The full deterministic search space for Phase 3C.

        Not run in the pilot; reserved for the full search.
        """
        return cls(
            name="default",
            parameter_values={
                "chunk_size": [100, 150, 200, 300, 400, 500],
                "chunk_overlap": [10, 20, 40, 50, 75],
                "top_k": [2, 3, 4, 5, 6],
            },
        )

    @classmethod
    def pilot(cls) -> "SearchSpace":
        """The small deterministic space used by ``--pilot``.

        2 x 2 x 2 = 8 configurations.
        """
        return cls(
            name="pilot",
            parameter_values={
                "chunk_size": [100, 300],
                "chunk_overlap": [20, 50],
                "top_k": [3, 5],
            },
        )

    @classmethod
    def adaptive_pilot(cls) -> "SearchSpace":
        """Larger search space for Phase 3D adaptive search pilot.

        7 x 5 x 5 = 175 possible configurations.
        """
        return cls(
            name="adaptive-pilot",
            parameter_values={
                "chunk_size": [100, 150, 200, 250, 300, 400, 500],
                "chunk_overlap": [10, 20, 40, 50, 75],
                "top_k": [2, 3, 4, 5, 6],
            },
        )