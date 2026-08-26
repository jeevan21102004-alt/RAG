"""RL environment for retrieval configuration selection (Phase 4).

``RetrievalOptimizationEnvironment`` frames budget-limited configuration
selection as a sequential decision problem::

    state (optimization history)
        -> action (configuration ID in 0..N-1)
        -> run ONE local retrieval experiment
        -> reward (objective + improvement + exploration)
        -> next state

The environment runs experiments through the existing ``run_experiment``
pipeline with a generation-disabled adapter, so **no Gemini calls and no
internet access** occur.

Action IDs map deterministically to configurations: action ``i`` is the
``i``-th combination produced by
``SearchSpace.generate_parameter_combinations()`` (itertools.product order,
ascending values).  For the pilot space that is exactly ``0 .. 174``.

Invalid actions are **rejected with an explanatory error** — the
environment never silently substitutes a different configuration:

- unknown action ID            -> :class:`InvalidActionError`
- already-evaluated config     -> :class:`AlreadyEvaluatedError`
- step after episode end       -> :class:`EpisodeDoneError`

The policy's invalid-action mask is an optimization; the environment
validates independently.
"""

from __future__ import annotations

from typing import Any

from .adaptive_search import get_neighbors, param_key
from .evaluation_schema import EvaluationCase
from .experiment_config import ExperimentConfig
from .experiment_runner import run_experiment
from .objective import ObjectiveConfig, calculate_objective, normalize_latency
from .optimization_reward import (
    DEFAULT_EXPLORATION_BONUS,
    DEFAULT_IMPROVEMENT_WEIGHT,
    calculate_optimization_reward,
)
from .optimization_state import (
    OPTIMIZATION_STATE_DIM,
    OptimizationState,
    _normalize_index,
    optimization_state_to_vector,
)
from .search_space import SearchSpace
from .system_adapter import SystemAdapter


class InvalidActionError(ValueError):
    """Raised when an action ID is outside the valid range."""


class AlreadyEvaluatedError(ValueError):
    """Raised when the selected configuration was evaluated earlier."""


class EpisodeDoneError(ValueError):
    """Raised when step() is called after the episode terminated."""


class RetrievalOptimizationEnvironment:
    """Budget-limited RL environment over a retrieval search space."""

    def __init__(
        self,
        search_space: SearchSpace,
        cases: list[EvaluationCase],
        adapter: SystemAdapter,
        budget: int = 8,
        objective_config: ObjectiveConfig | None = None,
        improvement_weight: float = DEFAULT_IMPROVEMENT_WEIGHT,
        exploration_bonus: float = DEFAULT_EXPLORATION_BONUS,
        delay: float = 0.0,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if not cases:
            raise ValueError("at least one evaluation case is required")
        search_space.raise_if_invalid()

        self.search_space = search_space
        self.cases = cases
        self.adapter = adapter
        self.budget = int(budget)
        self.objective_config = objective_config or ObjectiveConfig()
        self.improvement_weight = float(improvement_weight)
        self.exploration_bonus = float(exploration_bonus)
        self.delay = delay

        # Deterministic action-ID mapping: index in product order.
        self._combinations: list[dict[str, int]] = list(
            search_space.generate_parameter_combinations()
        )
        width = len(str(len(self._combinations)))
        self._configs: dict[int, ExperimentConfig] = {}
        for action_id, combo in enumerate(self._combinations):
            parts = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
            self._configs[action_id] = ExperimentConfig(
                experiment_id=f"opt-{action_id:0{width}d}",
                name=f"RL {action_id} ({parts})",
                description=f"RL-selected configuration with {parts}.",
                parameters=dict(combo),
                tags=["rl_optimization", search_space.name],
            )

        self.action_space_size = len(self._combinations)
        self.reset()

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def action_to_config(self, action: int) -> ExperimentConfig:
        """Deterministically map an action ID to its ``ExperimentConfig``."""
        if not isinstance(action, int) or isinstance(action, bool):
            raise InvalidActionError(
                f"Action must be an integer ID in [0, {self.action_space_size - 1}]; "
                f"got {action!r}."
            )
        if action < 0 or action >= self.action_space_size:
            raise InvalidActionError(
                f"Action ID {action} is out of range "
                f"[0, {self.action_space_size - 1}]."
            )
        return self._configs[action]

    def valid_action_mask(self) -> list[bool]:
        """Boolean mask; ``True`` marks actions that are still selectable."""
        evaluated = set(self._evaluated.keys())
        mask = []
        for combo in self._combinations:
            mask.append(param_key(combo) not in evaluated)
        return mask

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: int) -> ExperimentConfig:
        """Validate *action*, raising a specific error when invalid."""
        if self.done:
            raise EpisodeDoneError(
                "Episode already terminated; no further actions are accepted."
            )
        config = self.action_to_config(action)
        key = param_key(dict(config.parameters))
        if key in self._evaluated:
            raise AlreadyEvaluatedError(
                f"Configuration for action {action} "
                f"({config.experiment_id}) was already evaluated."
            )
        return config

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(self) -> OptimizationState:
        """Build the deterministic :class:`OptimizationState` snapshot."""
        successful = [
            m for m in self._evaluated.values() if m["status"] == "SUCCESS"
        ]
        failed = len(self._evaluated) - len(successful)

        objectives = [
            m["objective"] for m in successful if m["objective"] is not None
        ]
        f1s = [m["f1"] for m in successful if m["f1"] is not None]
        ctxs = [m["context_relevance"] for m in successful if m["context_relevance"] is not None]
        lat_scores = [m["latency_score"] for m in successful if m["latency_score"] is not None]

        values = self.search_space.parameter_values
        prev_params = self.last_action_parameters or {}

        return OptimizationState(
            best_objective=self.current_best_objective or 0.0,
            best_f1=max(f1s) if f1s else 0.0,
            best_context_relevance=max(ctxs) if ctxs else 0.0,
            best_latency_score=max(lat_scores) if lat_scores else 0.0,
            budget_consumed_fraction=len(self._order) / self.budget,
            space_evaluated_fraction=(
                len(self._evaluated) / self.action_space_size
                if self.action_space_size
                else 0.0
            ),
            successful_fraction=min(1.0, len(successful) / self.budget),
            failed_fraction=min(1.0, failed / self.budget),
            average_objective=(
                sum(objectives) / len(objectives) if objectives else 0.0
            ),
            last_improvement=max(0.0, self._last_step_improvement),
            unexplored_fraction=1.0 - (
                len(self._evaluated) / self.action_space_size
                if self.action_space_size
                else 1.0
            ),
            last_action_improved=1.0 if self._last_action_improved else 0.0,
            prev_chunk_size=_normalize_index(
                prev_params.get("chunk_size"), list(values.get("chunk_size", []))
            ),
            prev_chunk_overlap=_normalize_index(
                prev_params.get("chunk_overlap"),
                list(values.get("chunk_overlap", [])),
            ),
            prev_top_k=_normalize_index(
                prev_params.get("top_k"), list(values.get("top_k", []))
            ),
        )

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> OptimizationState:
        """Reset the episode and return the initial state."""
        # param_key -> metrics for evaluated configurations.
        self._evaluated: dict[tuple, dict[str, Any]] = {}
        self._order: list[dict[str, Any]] = []
        self.evaluations_used = 0
        self.current_best_objective: float | None = None
        self.previous_best_objective: float | None = None
        self.best_action_id: int | None = None
        self.last_action_id: int | None = None
        self.last_action_parameters: dict[str, int] | None = None
        self._last_step_improvement: float = 0.0
        self._last_action_improved: bool = False
        self.cumulative_reward: float = 0.0
        self.done = False
        return self._build_state()

    def _region_unexplored(self, parameters: dict[str, int]) -> bool:
        """True when neither this config nor any neighbour was evaluated."""
        if param_key(parameters) in self._evaluated:
            return False
        for neighbor in get_neighbors(parameters, self.search_space):
            if param_key(neighbor) in self._evaluated:
                return False
        return True

    def step(
        self, action: int
    ) -> tuple[OptimizationState, float, bool, dict[str, Any]]:
        """Validate *action*, run its experiment; return a transition."""
        config = self._validate_action(action)
        parameters = dict(config.parameters)

        previous_best = self.current_best_objective
        region_unexplored = self._region_unexplored(parameters)

        try:
            exp_result = run_experiment(
                config,
                self.cases,
                self.adapter,
                delay_between_questions=self.delay,
            )
            avg_scores = exp_result.diagnostics.get("average_scores", {}) or {}
            f1 = avg_scores.get("retrieval_f1")
            if f1 is None:
                f1 = exp_result.average_retrieval_score
            context_relevance = avg_scores.get("context_relevance")
            latency_ms = avg_scores.get("latency_ms")
            if latency_ms is None:
                latency_ms = exp_result.average_latency_ms
            status = "SUCCESS"
            api_errors = sum(
                1
                for record in exp_result.metadata.get("evaluation_results", [])
                if record.get("status") == "API_ERROR"
            )
        except Exception as error:  # noqa: BLE001 - configuration failure
            f1 = None
            context_relevance = None
            latency_ms = None
            status = f"CONFIG_ERROR: {error}"
            api_errors = 0

        objective = None
        latency_score = None
        if status == "SUCCESS":
            objective_result = calculate_objective(
                retrieval_f1=f1,
                context_relevance=context_relevance,
                latency_ms=latency_ms,
                weights=self.objective_config.weights,
                reference_latency_ms=self.objective_config.reference_latency_ms,
            )
            objective = objective_result.objective_score
            latency_score = normalize_latency(
                latency_ms, self.objective_config.reference_latency_ms
            )

        reward_result = calculate_optimization_reward(
            objective_score=objective,
            previous_best_objective=previous_best,
            region_unexplored=region_unexplored,
            improvement_weight=self.improvement_weight,
            exploration_bonus=self.exploration_bonus,
        )

        # ---- update optimization history ------------------------------
        self.previous_best_objective = previous_best
        if objective is not None and (
            self.current_best_objective is None
            or objective > self.current_best_objective
        ):
            self._last_step_improvement = (
                objective - previous_best
                if previous_best is not None
                else objective
            )
            self.current_best_objective = objective
            self.best_action_id = action
            self._last_action_improved = True
        else:
            self._last_step_improvement = 0.0
            self._last_action_improved = False

        metrics = {
            "status": status,
            "objective": objective,
            "f1": f1,
            "context_relevance": context_relevance,
            "latency_ms": latency_ms,
            "latency_score": latency_score,
            "api_errors": api_errors,
        }
        key = param_key(parameters)
        self._evaluated[key] = metrics
        record = {
            "action_id": action,
            "experiment_id": config.experiment_id,
            "parameters": parameters,
            "reward": reward_result.final_reward,
            **metrics,
        }
        self._order.append(record)
        self.evaluations_used += 1
        self.last_action_id = action
        self.last_action_parameters = parameters
        self.cumulative_reward += reward_result.final_reward

        # ---- termination ----------------------------------------------
        all_evaluated = len(self._evaluated) >= self.action_space_size
        budget_exhausted = self.evaluations_used >= self.budget
        self.done = budget_exhausted or all_evaluated

        next_state = self._build_state()
        metadata = {
            "action_id": action,
            "experiment_id": config.experiment_id,
            "parameters": parameters,
            "status": status,
            "objective": objective,
            "retrieval_f1": f1,
            "context_relevance": context_relevance,
            "latency_ms": latency_ms,
            "region_unexplored": region_unexplored,
            "reward_breakdown": reward_result.to_dict(),
            "evaluations_used": self.evaluations_used,
            "budget": self.budget,
            "current_best_objective": self.current_best_objective,
            "best_action_id": self.best_action_id,
            "cumulative_reward": self.cumulative_reward,
            "termination_reason": (
                "all_configurations_evaluated"
                if all_evaluated
                else ("budget_exhausted" if budget_exhausted else "none")
            ),
        }
        return next_state, reward_result.final_reward, self.done, metadata