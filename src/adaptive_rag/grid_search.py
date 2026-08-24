"""Deterministic grid search engine for automated configuration search.

Phase 3C performs an **exhaustive, sequential grid search** over a manually
defined :class:`SearchSpace`.  Every configuration is run through the
existing experiment runner and retrieval benchmark adapter, evaluated with
the existing evaluation engine, scored with the objective function, and
ranked.

There is **no parallelization**, **no randomness**, and **no external
optimization library**.

Failure handling distinguishes:

- **CONFIGURATION FAILURE** — an individual configuration could not be
  evaluated (e.g. its experiment raised).  It is recorded and the search
  continues with the next configuration.
- **SYSTEM FAILURE** — the benchmark infrastructure itself is broken (e.g.
  the dataset or corpus is missing, or every configuration fails with the
  same infrastructure error).  The search stops and reports it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_generator import generate_configurations
from .evaluation_schema import EvaluationCase
from .experiment_config import ExperimentConfig
from .experiment_runner import run_experiment
from .objective import (
    ObjectiveConfig,
    calculate_objective,
)
from .retrieval_benchmark_adapter import RetrievalBenchmarkAdapter
from .search_result import SearchRankingEntry, SearchResult
from .search_space import SearchSpace
from .system_adapter import SystemAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_BENCHMARK_DATASET_PATH = PROJECT_ROOT / "data" / "retrieval_benchmark_questions.json"
ENTERPRISE_DATASET_PATH = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"


class SearchSystemError(RuntimeError):
    """Raised when the search infrastructure itself fails."""


def load_search_cases(
    dataset: str = "enterprise",
    max_questions: int | None = None,
) -> list[EvaluationCase]:
    """Load benchmark questions deterministically.

    ``dataset == "enterprise"`` uses the 20-question enterprise benchmark;
    ``dataset == "default"`` uses the original 8-question benchmark.  When
    *max_questions* is given, only the first N questions are used (the
    same N for every configuration).
    """
    if dataset == "enterprise":
        path = ENTERPRISE_DATASET_PATH
    elif dataset == "default":
        path = RETRIEVAL_BENCHMARK_DATASET_PATH
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if not path.exists():
        raise SearchSystemError(f"Dataset file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if max_questions is not None:
        data = data[:max_questions]
    return [EvaluationCase(**item) for item in data]


class GridSearchEngine:
    """Deterministic sequential grid search over a :class:`SearchSpace`.

    Parameters
    ----------
    search_space:
        The space of parameter combinations to evaluate.
    dataset:
        ``"enterprise"`` or ``"default"`` (which question set to use).
    max_questions:
        Optional deterministic cap on questions per configuration.
    adapter:
        Adapter to evaluate.  Defaults to a generation-disabled
        :class:`RetrievalBenchmarkAdapter` (no Gemini calls).
    objective_config:
        Objective weights and latency reference.  Defaults preserved.
    delay:
        Delay between questions (rate-limit protection; 0 for API-free
        runs and tests).
    search_id:
        Optional stable identifier.  Defaults to a timestamped ID.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        dataset: str = "enterprise",
        max_questions: int | None = None,
        adapter: SystemAdapter | None = None,
        objective_config: ObjectiveConfig | None = None,
        delay: float = 0.0,
        search_id: str | None = None,
    ) -> None:
        self.search_space = search_space
        self.dataset = dataset
        self.max_questions = max_questions
        self.delay = delay
        if adapter is None:
            data_dir = ENTERPRISE_KB_DIR if dataset == "enterprise" else None
            adapter = RetrievalBenchmarkAdapter(
                enable_generation=False, data_dir=data_dir
            )
        self.adapter = adapter
        self.objective_config = objective_config or ObjectiveConfig()
        self.search_id = search_id or self._default_search_id()

    def _default_search_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"gridsearch_{self.search_space.name}_{stamp}"
# ------------------------------------------------------------------
    # Search execution
    # ------------------------------------------------------------------

    def run(self) -> SearchResult:
        """Execute the full deterministic grid search.

        Returns a :class:`SearchResult` with every evaluated configuration
        and the ranked best configuration.

        Raises :class:`SearchSystemError` when the infrastructure itself is
        broken (missing dataset / corpus, or every configuration failing with
        the same error).

        Individual configuration failures are recorded and do not stop the
        search.
        """
        # ---- infrastructure validation (SYSTEM FAILURE) ----------------
        try:
            cases = load_search_cases(self.dataset, self.max_questions)
        except ValueError as error:
            raise SearchSystemError(f"Invalid dataset: {error}") from error
        if not cases:
            raise SearchSystemError(
                "No evaluation cases could be loaded for dataset "
                f"{self.dataset!r}."
            )
        if self.dataset == "enterprise" and not ENTERPRISE_KB_DIR.exists():
            raise SearchSystemError(
                f"Enterprise KB directory missing: {ENTERPRISE_KB_DIR}"
            )

        configurations = generate_configurations(self.search_space, prefix="grid")
        entries: list[SearchRankingEntry] = []
        config_failures: list[tuple[ExperimentConfig, Exception]] = []
        api_errors_total = 0

        for config in configurations:
            try:
                exp_result = run_experiment(
                    config,
                    cases,
                    self.adapter,
                    delay_between_questions=self.delay,
                )
            except Exception as error:  # noqa: BLE001 - config failure
                config_failures.append((config, error))
                entries.append(self._failure_entry(config, str(error)))
                continue
# Extract per-configuration metrics from the experiment result.
            metrics = self._extract_metrics(exp_result)
            api_errors_total += metrics["api_errors"]

            objective_result = calculate_objective(
                retrieval_f1=metrics["retrieval_f1"],
                context_relevance=metrics["context_relevance"],
                latency_ms=metrics["latency_ms"],
                weights=self.objective_config.weights,
                reference_latency_ms=self.objective_config.reference_latency_ms,
            )

            entries.append(
                SearchRankingEntry(
                    experiment_id=config.experiment_id,
                    parameters=dict(config.parameters),
                    objective_score=objective_result.objective_score,
                    retrieval_f1=metrics["retrieval_f1"],
                    context_relevance=metrics["context_relevance"],
                    latency_ms=metrics["latency_ms"],
                    status="SUCCESS",
                )
            )

        # If every configuration failed with the same infrastructure cause,
        # treat it as a SYSTEM failure rather than N configuration failures.
        if entries and all(e.status.startswith("CONFIG_ERROR") for e in entries):
            first_error = config_failures[0][1] if config_failures else None
            raise SearchSystemError(
                "Every configuration failed during evaluation: "
                f"{first_error}"
            )

        completed = [e for e in entries if e.status == "SUCCESS"]
        ranked = self._rank_entries(entries)
        best = ranked[0] if ranked and ranked[0].status == "SUCCESS" else None

        return SearchResult(
            search_id=self.search_id,
            total_configurations=len(configurations),
            completed_configurations=len(completed),
            failed_configurations=len(config_failures),
            ranking=ranked,
            best_configuration=best,
            best_objective_score=best.objective_score if best else None,
            objective_definition={
                "weights": dict(self.objective_config.weights),
                "reference_latency_ms": self.objective_config.reference_latency_ms,
            },
            metadata={
                "search_space": self.search_space.to_dict(),
                "dataset": self.dataset,
                "questions_per_config": len(cases),
                "total_evaluations": len(configurations) * len(cases),
                "api_errors_total": api_errors_total,
                "generation_enabled": False,
                "config_failures": [str(err) for _, err in config_failures],
            },
        )
# ------------------------------------------------------------------
    # Metrics + ranking helpers
    # ------------------------------------------------------------------

    def _extract_metrics(self, result: Any) -> dict[str, float | int | None]:
        """Extract objective inputs from an ``ExperimentResult``.

        Uses the diagnostic engine's aggregated ``average_scores`` (which
        is populated by the existing evaluation pipeline) and falls back to
        the top-level fields when a key is absent.
        """
        avg_scores = result.diagnostics.get("average_scores", {}) or {}

        retrieval_f1 = avg_scores.get("retrieval_f1")
        if retrieval_f1 is None:
            retrieval_f1 = result.average_retrieval_score

        context_relevance = avg_scores.get("context_relevance")
        latency_ms = avg_scores.get("latency_ms")
        if latency_ms is None:
            latency_ms = result.average_latency_ms

        api_errors = 0
        for record in result.metadata.get("evaluation_results", []):
            if record.get("status") == "API_ERROR":
                api_errors += 1

        return {
            "retrieval_f1": retrieval_f1,
            "context_relevance": context_relevance,
            "latency_ms": latency_ms,
            "api_errors": api_errors,
        }

    def _failure_entry(
        self,
        config: ExperimentConfig,
        error: str,
    ) -> SearchRankingEntry:
        """Build a ranking entry for a configuration that failed to run."""
        return SearchRankingEntry(
            experiment_id=config.experiment_id,
            parameters=dict(config.parameters),
            objective_score=None,
            retrieval_f1=None,
            context_relevance=None,
            latency_ms=None,
            status=f"CONFIG_ERROR: {error}",
        )

    def _rank_entries(
        self,
        entries: list[SearchRankingEntry],
    ) -> list[SearchRankingEntry]:
        """Rank entries by objective score (descending), deterministically.

        Configurations without a score (failures) are ranked last.
        Ties are broken by experiment_id for a fixed, reproducible order.
        """
        return rank_entries(entries)


def rank_entries(entries: list[SearchRankingEntry]) -> list[SearchRankingEntry]:
    """Rank entries deterministically by objective score (descending).

    - Entries with a ``None`` objective score sort last.
    - Entries with equal scores are ordered by ``experiment_id`` so the
      ranking is fully reproducible.
    """
    def _sort_key(entry: SearchRankingEntry) -> tuple[int, float | None, str]:
        has_score = entry.objective_score is not None
        score = entry.objective_score if has_score else -1.0
        return (1 if has_score else 0, score, entry.experiment_id)

    return sorted(entries, key=_sort_key, reverse=True)
