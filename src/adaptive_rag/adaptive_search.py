"""Deterministic adaptive configuration search (Phase 3D).

This module implements a **lightweight, transparent, heuristic** adaptive
search strategy that uses previous experiment results to decide which
configuration to evaluate next.

Key difference from grid search (Phase 3C)::

    GRID SEARCH:   generate all -> evaluate all -> rank
    ADAPTIVE:      generate candidates -> evaluate some -> use results ->
                   select promising neighbour -> evaluate -> repeat

**This is a deterministic heuristic adaptive search strategy.**

It is NOT Bayesian optimization, reinforcement learning, or a guarantee of
global optimality.  There is no randomness, no neural network, and no
external optimization library — only the Python standard library.

Selection heuristic (documented, deterministic)::

    priority(c) = neighbour_average_objective(c) + exploration_bonus
                  if c has at least one evaluated neighbour

    priority(c) = exploration_bonus
                  if c has no evaluated neighbours

Candidates with higher priority are evaluated first.  Ties are broken by
the deterministic parameter key so the search is fully reproducible.
"""

from __future__ import annotations

from typing import Any

from .adaptive_search_result import (
    AdaptiveRankingEntry,
    AdaptiveSearchResult,
    EvaluationOrderEntry,
)
from .evaluation_schema import EvaluationCase
from .experiment_config import ExperimentConfig
from .experiment_runner import run_experiment
from .grid_search import SearchSystemError, load_search_cases
from .objective import ObjectiveConfig, calculate_objective
from .search_result import SearchRankingEntry
from .search_space import SearchSpace
from .system_adapter import SystemAdapter

DEFAULT_EXPLORATION_BONUS: float = 0.05
DEFAULT_MAX_CONFIGURATIONS: int = 8


def param_key(parameters: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """Deterministic hashable key for a parameter dictionary."""
    return tuple(sorted(parameters.items()))


def select_initial_combinations(
    space: SearchSpace,
) -> list[dict[str, int]]:
    """Select the deterministic initial exploration set.

    Strategy (documented in ``experiments/adaptive_search_methodology.md``):

    1. min-min-min corner   (smallest chunk_size / overlap / top_k)
    2. max-max-max corner   (largest chunk_size / overlap / top_k)
    3. opposite corner A    (max chunk_size, min overlap, min top_k)
    4. opposite corner B    (min chunk_size, max overlap, max top_k)
    5. centre               (middle value of every parameter list)

    Duplicates are removed while preserving the order above, so the same
    search space always yields the same initial set in the same order.
    """
    space.raise_if_invalid()
    values = {k: list(v) for k, v in space.parameter_values.items()}
    keys = list(values)

    def combo(*idx: int) -> dict[str, int]:
        return {k: values[k][i] for k, i in zip(keys, idx)}

    mid = len(keys) // 2
    candidates = [
        combo(0, 0, 0),
        combo(-1, -1, -1),
        combo(-1, 0, 0),
        combo(0, -1, -1),
    ]
    # Centre configuration: middle index of each parameter list.
    candidates.append({k: values[k][len(values[k]) // 2] for k in keys})
    _ = mid  # centre computed directly above

    seen: set[tuple] = set()
    ordered: list[dict[str, int]] = []
    for candidate in candidates:
        key = param_key(candidate)
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def get_neighbors(
    parameters: dict[str, int],
    space: SearchSpace,
) -> list[dict[str, int]]:
    """Return valid configurations differing by ONE parameter step.

    For each parameter, stepping one position up or down its sorted value
    list produces a neighbour.  Only combinations that exist in the search
    space are returned; duplicates are removed deterministically.
    """
    neighbors: list[dict[str, int]] = []
    seen: set[tuple] = set()
    for key, values in space.parameter_values.items():
        try:
            index = values.index(parameters[key])
        except (KeyError, ValueError):
            continue
        for step in (-1, 1):
            new_index = index + step
            if 0 <= new_index < len(values):
                candidate = dict(parameters)
                candidate[key] = values[new_index]
                candidate_key = param_key(candidate)
                if candidate_key not in seen:
                    seen.add(candidate_key)
                    neighbors.append(candidate)
    return neighbors


def _extract_metrics(result: Any) -> dict[str, float | int | None]:
    """Extract objective inputs from an ``ExperimentResult``.

    Mirrors the extraction used by grid search: reads the diagnostic
    engine's aggregated ``average_scores`` and falls back to top-level
    fields when a key is absent.
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


class AdaptiveSearchEngine:
    """Deterministic heuristic adaptive search over a ``SearchSpace``.

    The engine decides **WHAT TO RUN NEXT**.  It does NOT implement how
    experiments run — that is delegated to :func:`run_experiment`, the
    adapter, and the existing evaluation pipeline.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        dataset: str = "enterprise",
        max_questions: int | None = None,
        adapter: SystemAdapter | None = None,
        objective_config: ObjectiveConfig | None = None,
        max_configurations: int = DEFAULT_MAX_CONFIGURATIONS,
        exploration_bonus: float = DEFAULT_EXPLORATION_BONUS,
        delay: float = 0.0,
        search_id: str | None = None,
    ) -> None:
        self.search_space = search_space
        self.dataset = dataset
        self.max_questions = max_questions
        if adapter is None:
            from .grid_search import ENTERPRISE_KB_DIR
            from .retrieval_benchmark_adapter import RetrievalBenchmarkAdapter

            data_dir = ENTERPRISE_KB_DIR if dataset == "enterprise" else None
            adapter = RetrievalBenchmarkAdapter(
                enable_generation=False, data_dir=data_dir
            )
        self.adapter = adapter
        self.objective_config = objective_config or ObjectiveConfig()
        self.max_configurations = max(1, int(max_configurations))
        self.exploration_bonus = float(exploration_bonus)
        self.delay = delay
        self.search_id = search_id or self._default_search_id()

        # Full deterministic enumeration of the space.
        self._all_combinations: list[dict[str, int]] = list(
            self.search_space.generate_parameter_combinations()
        )
        self._combo_ids: dict[tuple, str] = {}
        width = len(str(len(self._all_combinations)))
        for index, combo in enumerate(self._all_combinations, start=1):
            self._combo_ids[param_key(combo)] = f"adaptive-{index:0{width}d}"

        # Observed state keyed by param_key.
        self._metrics: dict[tuple, dict[str, float | int | None]] = {}
        self._objectives: dict[tuple, float | None] = {}
        self._statuses: dict[tuple, str] = {}
        self._order_entries: list[EvaluationOrderEntry] = []
        self._ranking_entries: list[AdaptiveRankingEntry] = []

    def _default_search_id(self) -> str:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"adaptivesearch_{self.search_space.name}_{stamp}"

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _neighbor_objective_average(
        self,
        parameters: dict[str, int],
    ) -> tuple[float | None, int]:
        """Average objective over *evaluated* neighbours.

        Returns ``(average, count)``; average is ``None`` when no
        evaluated neighbour has a computed objective (failures excluded).
        """
        objectives: list[float] = []
        for neighbor in get_neighbors(parameters, self.search_space):
            key = param_key(neighbor)
            if key in self._objectives and self._objectives[key] is not None:
                objectives.append(float(self._objectives[key]))
        if not objectives:
            return None, 0
        return sum(objectives) / len(objectives), len(objectives)

    def _priority(self, parameters: dict[str, int]) -> tuple[float, str]:
        """Deterministic priority score plus a human-readable rationale."""
        neighbor_avg, neighbor_count = self._neighbor_objective_average(parameters)
        if neighbor_count == 0:
            reason = (
                f"exploration bonus: no evaluated neighbours; "
                f"priority {self.exploration_bonus:.4f}"
            )
            return self.exploration_bonus, reason
        priority = neighbor_avg + self.exploration_bonus
        reason = (
            f"selected highest-priority unevaluated candidate: "
            f"neighbour average objective {neighbor_avg:.4f} "
            f"over {neighbor_count} neighbour(s) "
            f"(+ bonus {self.exploration_bonus:.4f}) "
            f"= priority {priority:.4f}"
        )
        return priority, reason

    def _select_next(self) -> tuple[dict[str, int], str]:
        """Pick the unevaluated candidate with the highest priority.

        Ties are broken by the deterministic parameter key so repeated
        runs select configurations in exactly the same order.
        """
        best_key: tuple | None = None
        best_priority = float("-inf")
        best_reason = ""
        for combo in self._all_combinations:
            key = param_key(combo)
            if key in self._statuses:
                continue  # already evaluated (success or failure)
            priority, reason = self._priority(combo)
            if (
                best_key is None
                or priority > best_priority
                or (priority == best_priority and key < best_key)
            ):
                best_key, best_priority, best_reason = key, priority, reason
        if best_key is None:
            raise SearchSystemError("No unevaluated candidate remains.")
        for combo in self._all_combinations:
            if param_key(combo) == best_key:
                return combo, best_reason
        raise SearchSystemError("Internal selection error.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _build_config(self, parameters: dict[str, int]) -> ExperimentConfig:
        experiment_id = self._combo_ids[param_key(parameters)]
        parts = ", ".join(f"{k}={v}" for k, v in sorted(parameters.items()))
        return ExperimentConfig(
            experiment_id=experiment_id,
            name=f"Adaptive {experiment_id} ({parts})",
            description=f"Adaptive-search configuration with {parts}.",
            parameters=dict(parameters),
            tags=["adaptive_search", self.search_space.name],
        )

    def _evaluate(
        self,
        parameters: dict[str, int],
        cases: list[EvaluationCase],
        reason: str,
    ) -> AdaptiveRankingEntry:
        """Evaluate one configuration and record the outcome.

        Configuration-level failures are recorded (never fabricated as
        metrics) and do not stop the search.
        """
        config = self._build_config(parameters)
        key = param_key(parameters)
        try:
            exp_result = run_experiment(
                config,
                cases,
                self.adapter,
                delay_between_questions=self.delay,
            )
            metrics = _extract_metrics(exp_result)
            objective_result = calculate_objective(
                retrieval_f1=metrics["retrieval_f1"],
                context_relevance=metrics["context_relevance"],
                latency_ms=metrics["latency_ms"],
                weights=self.objective_config.weights,
                reference_latency_ms=self.objective_config.reference_latency_ms,
            )
            objective = objective_result.objective_score
            status = "SUCCESS"
        except Exception as error:  # noqa: BLE001 - configuration failure
            metrics = {
                "retrieval_f1": None,
                "context_relevance": None,
                "latency_ms": None,
                "api_errors": 0,
            }
            objective = None
            status = f"CONFIG_ERROR: {error}"

        entry = AdaptiveRankingEntry(
            experiment_id=config.experiment_id,
            parameters=dict(parameters),
            objective_score=objective,
            retrieval_f1=metrics["retrieval_f1"],
            context_relevance=metrics["context_relevance"],
            latency_ms=metrics["latency_ms"],
            status=status,
        )
        self._metrics[key] = metrics
        self._objectives[key] = objective
        self._statuses[key] = status
        self._ranking_entries.append(entry)
        self._order_entries.append(
            EvaluationOrderEntry(
                experiment_id=config.experiment_id,
                parameters=dict(parameters),
                objective_score=objective,
                retrieval_f1=metrics["retrieval_f1"],
                context_relevance=metrics["context_relevance"],
                latency_ms=metrics["latency_ms"],
                status=status,
                selection_reason=reason,
            )
        )
        return entry

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> AdaptiveSearchResult:
        """Execute the deterministic adaptive search.

        1. Deterministic initial exploration (corners + centre).
        2. Repeatedly select the highest-priority unevaluated candidate
           using the neighbour-average heuristic until the configuration
           budget is exhausted or the space is fully explored.

        Raises :class:`SearchSystemError` when the infrastructure itself
        is broken.  Individual configuration failures are recorded.
        """
        try:
            cases = load_search_cases(self.dataset, self.max_questions)
        except ValueError as error:
            raise SearchSystemError(f"Invalid dataset: {error}") from error
        if not cases:
            raise SearchSystemError(
                f"No evaluation cases could be loaded for dataset "
                f"{self.dataset!r}."
            )
        from .grid_search import ENTERPRISE_KB_DIR

        if self.dataset == "enterprise" and not ENTERPRISE_KB_DIR.exists():
            raise SearchSystemError(
                f"Enterprise KB directory missing: {ENTERPRISE_KB_DIR}"
            )

        initial_reason = (
            "initial exploration: deterministic corner/centre of the "
            "search space"
        )

        # Phase 1 — baseline exploration, capped by the budget.
        for combo in select_initial_combinations(self.search_space):
            if len(self._statuses) >= self.max_configurations:
                break
            self._evaluate(combo, cases, initial_reason)

        # Phase 2 — adaptive selection until the budget is exhausted.
        while len(self._statuses) < self.max_configurations:
            unevaluated_left = any(
                param_key(c) not in self._statuses
                for c in self._all_combinations
            )
            if not unevaluated_left:
                break
            combo, reason = self._select_next()
            self._evaluate(combo, cases, reason)

        ranked = self._rank_entries(list(self._ranking_entries))
        best = ranked[0] if ranked and ranked[0].status == "SUCCESS" else None
        best_score = best.objective_score if best is not None else None

        evaluated = len(self._order_entries)
        failed = sum(
            1 for e in self._ranking_entries if e.status != "SUCCESS"
        )
        api_errors_total = sum(
            int(m.get("api_errors") or 0) for m in self._metrics.values()
        )

        return AdaptiveSearchResult(
            search_id=self.search_id,
            search_space=self.search_space.to_dict(),
            total_possible_configurations=len(self._all_combinations),
            configurations_evaluated=evaluated,
            configurations_remaining=len(self._all_combinations) - evaluated,
            failed_configurations=failed,
            best_configuration=best,
            best_objective_score=best_score,
            ranking=ranked,
            evaluation_order=list(self._order_entries),
            selection_reasons={
                entry.experiment_id: entry.selection_reason
                for entry in self._order_entries
            },
            objective_definition={
                "weights": dict(self.objective_config.weights),
                "reference_latency_ms": self.objective_config.reference_latency_ms,
                "exploration_bonus": self.exploration_bonus,
            },
            metadata={
                "search_space": self.search_space.to_dict(),
                "dataset": self.dataset,
                "questions_per_config": len(cases),
                "max_configurations": self.max_configurations,
                "total_evaluations": evaluated * len(cases),
                "api_errors_total": api_errors_total,
                "generation_enabled": False,
                "strategy": "deterministic_neighbour_heuristic",
            },
        )

    def _rank_entries(
        self,
        entries: list[AdaptiveRankingEntry],
    ) -> list[AdaptiveRankingEntry]:
        """Rank entries by objective score (descending), deterministically.

        Entries without a score (failures) sort last; ties break on
        ``experiment_id`` so the ordering is fully reproducible.
        """

        def _sort_key(entry: AdaptiveRankingEntry) -> tuple[int, float, str]:
            has_score = entry.objective_score is not None
            score = float(entry.objective_score) if has_score else -1.0
            return (1 if has_score else 0, score, entry.experiment_id)

        return sorted(entries, key=_sort_key, reverse=True)