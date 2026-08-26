"""Baseline configuration-selection policies (Phase 4).

Implements random, grid, and adaptive baselines so they can be compared
fairly against the RL policy.  All four methods use the **same** search
space, budget, dataset, objective, and question set.

Each baseline produces an :class:`OptimizationResult` with the identical
``evaluation_order``/``best_*`` fields as the RL method.
"""

from __future__ import annotations

import random
from typing import Any

from .adaptive_search import AdaptiveSearchEngine
from .evaluation_schema import EvaluationCase
from .experiment_config import ExperimentConfig
from .experiment_runner import run_experiment
from .objective import ObjectiveConfig, calculate_objective
from .optimization_result import OptimizationEntry, OptimizationResult
from .search_space import SearchSpace
from .system_adapter import SystemAdapter


def _evaluate_combos(
    combos: list[dict[str, int]],
    adapter: SystemAdapter,
    cases: list[EvaluationCase],
    objective_config: ObjectiveConfig,
    method: str,
    seed: int | None,
    budget: int,
    search_space: SearchSpace,
) -> OptimizationResult:
    """Evaluate *combos* through the standard pipeline and aggregate.

    This is the **only** place baselines run experiments — reusing
    ``run_experiment``/``calculate_objective`` exactly as the RL method.
    """
    order: list[OptimizationEntry] = []
    best_objective: float | None = None
    best_f1: float | None = None
    best_ctx: float | None = None
    best_latency_ms: float | None = None
    best_entry: OptimizationEntry | None = None

    for step, params in enumerate(combos[:budget], start=1):
        config = ExperimentConfig(
            experiment_id=f"{method}-{step:03d}",
            name=f"{method} {step}",
            description="baseline selection",
            parameters=dict(params),
            tags=["baseline", method],
        )
        try:
            exp_result = run_experiment(
                config, cases, adapter, delay_between_questions=0.0
            )
            avg = exp_result.diagnostics.get("average_scores", {}) or {}
            f1 = avg.get("retrieval_f1") or exp_result.average_retrieval_score
            ctx = avg.get("context_relevance")
            latency_ms = avg.get("latency_ms") or exp_result.average_latency_ms
            status = "SUCCESS"
        except Exception as error:  # noqa: BLE001
            f1, ctx, latency_ms, status = None, None, None, f"CONFIG_ERROR: {error}"

        objective = None
        if status == "SUCCESS":
            r = calculate_objective(
                retrieval_f1=f1,
                context_relevance=ctx,
                latency_ms=latency_ms,
                weights=objective_config.weights,
                reference_latency_ms=objective_config.reference_latency_ms,
            )
            objective = r.objective_score

        entry = OptimizationEntry(
            method=method,
            step=step,
            action_id=None,
            parameters=dict(params),
            objective_score=objective,
            retrieval_f1=f1,
            context_relevance=ctx,
            latency_ms=latency_ms,
            status=status,
        )
        order.append(entry)
        if objective is not None and (
            best_objective is None or objective > best_objective
        ):
            best_objective = objective
            best_f1 = f1
            best_ctx = ctx
            best_latency_ms = latency_ms
            best_entry = entry

    return OptimizationResult(
        method=method,
        seed=seed,
        budget=budget,
        search_space=search_space.to_dict(),
        configurations_evaluated=len(order),
        evaluation_order=order,
        best_configuration=best_entry,
        best_objective=best_objective,
        best_f1=best_f1,
        best_context_relevance=best_ctx,
        best_latency_ms=best_latency_ms,
        cumulative_reward=None,
        metadata={"baseline": method},
    )


class RandomSearchBaseline:
    """Evaluates a fixed random subset of the search space."""

    def __init__(
        self,
        search_space: SearchSpace,
        budget: int,
        seed: int,
        adapter: SystemAdapter,
        cases: list[EvaluationCase],
        objective_config: ObjectiveConfig | None = None,
    ) -> None:
        self.search_space = search_space
        self.budget = int(budget)
        self.seed = seed
        self.adapter = adapter
        self.cases = cases
        self.objective_config = objective_config or ObjectiveConfig()

    def run(self) -> OptimizationResult:
        rng = random.Random(self.seed)
        combos = list(self.search_space.generate_parameter_combinations())
        rng.shuffle(combos)
        return _evaluate_combos(
            combos=combos[: self.budget],
            adapter=self.adapter,
            cases=self.cases,
            objective_config=self.objective_config,
            method="random",
            seed=self.seed,
            budget=self.budget,
            search_space=self.search_space,
        )


class GridSearchBaseline:
    """Selects the first N configurations in deterministic product order."""

    def __init__(
        self,
        search_space: SearchSpace,
        budget: int,
        adapter: SystemAdapter,
        cases: list[EvaluationCase],
        objective_config: ObjectiveConfig | None = None,
    ) -> None:
        self.search_space = search_space
        self.budget = int(budget)
        self.adapter = adapter
        self.cases = cases
        self.objective_config = objective_config or ObjectiveConfig()

    def run(self) -> OptimizationResult:
        combos = list(self.search_space.generate_parameter_combinations())
        return _evaluate_combos(
            combos=combos[: self.budget],
            adapter=self.adapter,
            cases=self.cases,
            objective_config=self.objective_config,
            method="grid",
            seed=None,
            budget=self.budget,
            search_space=self.search_space,
        )


class AdaptiveSearchBaseline:
    """Reuses the Phase 3D neighbour-heuristic adaptive engine.

    The engine performs deterministic corner/centre exploration followed by
    greedy neighbour exploitation, all within the same budget as the other
    methods.  The exact question set is passed through ``cases_override`` so
    every method evaluates identical questions.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        budget: int,
        adapter: SystemAdapter,
        cases: list[EvaluationCase],
        objective_config: ObjectiveConfig | None = None,
    ) -> None:
        self.search_space = search_space
        self.budget = int(budget)
        self.adapter = adapter
        self.cases = cases
        self.objective_config = objective_config or ObjectiveConfig()

    def run(self) -> OptimizationResult:
        engine = AdaptiveSearchEngine(
            self.search_space,
            dataset="enterprise",
            max_questions=None,
            adapter=self.adapter,
            objective_config=self.objective_config,
            max_configurations=self.budget,
            search_id="adaptive-baseline",
        )
        result = engine.run(cases_override=self.cases)

        order: list[OptimizationEntry] = []
        best_entry: OptimizationEntry | None = None
        best_obj = best_f1 = best_ctx = best_lat = None
        for step, entry in enumerate(result.evaluation_order, start=1):
            opt_entry = OptimizationEntry(
                method="adaptive",
                step=step,
                action_id=None,
                parameters=dict(entry.parameters),
                objective_score=entry.objective_score,
                retrieval_f1=entry.retrieval_f1,
                context_relevance=entry.context_relevance,
                latency_ms=entry.latency_ms,
                status=entry.status,
            )
            order.append(opt_entry)
            if entry.objective_score is not None and (
                best_obj is None or entry.objective_score > best_obj
            ):
                best_obj = entry.objective_score
                best_f1 = entry.retrieval_f1
                best_ctx = entry.context_relevance
                best_lat = entry.latency_ms
                best_entry = opt_entry

        return OptimizationResult(
            method="adaptive",
            seed=None,
            budget=self.budget,
            search_space=self.search_space.to_dict(),
            configurations_evaluated=len(order),
            evaluation_order=order,
            best_configuration=best_entry,
            best_objective=best_obj,
            best_f1=best_f1,
            best_context_relevance=best_ctx,
            best_latency_ms=best_lat,
            cumulative_reward=None,
            metadata={
                "baseline": "adaptive",
                "strategy": "neighbour_heuristic",
            },
        )

