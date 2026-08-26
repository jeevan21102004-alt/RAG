"""Reward function for RL configuration selection (Phase 4).

This reward is **different from the legacy SEARCH/ANSWER reward** in
``src/adaptive_rag/reward.py`` — that module and the earlier Phase 7
experiments remain untouched.

Transparent default formula::

    base_reward        = objective_score                  (in [0, 1])
    improvement_bonus  = max(0, objective - previous_best)
    exploration_bonus  = bonus if the action's region was unexplored

    final_reward = base_reward
                 + improvement_weight * improvement_bonus
                 + exploration_bonus          (when applicable)

Defaults: ``improvement_weight = 0.5``, ``exploration_bonus = 0.05``
(applied when neither the chosen configuration nor any of its immediate
neighbours had been evaluated before this step).  Failed evaluations
(objective ``None``) receive reward ``0.0`` with every component ``None``
— failures are never fabricated into positive rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_IMPROVEMENT_WEIGHT: float = 0.5
DEFAULT_EXPLORATION_BONUS: float = 0.05


@dataclass(frozen=True)
class OptimizationRewardResult:
    """Component breakdown of one step's reward."""

    final_reward: float
    base_reward: float | None
    improvement_bonus: float | None
    exploration_component: float
    previous_best_objective: float | None
    new_best_objective: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_reward": self.final_reward,
            "base_reward": self.base_reward,
            "improvement_bonus": self.improvement_bonus,
            "exploration_component": self.exploration_component,
            "previous_best_objective": self.previous_best_objective,
            "new_best_objective": self.new_best_objective,
        }


def calculate_optimization_reward(
    objective_score: float | None,
    previous_best_objective: float | None,
    region_unexplored: bool,
    improvement_weight: float = DEFAULT_IMPROVEMENT_WEIGHT,
    exploration_bonus: float = DEFAULT_EXPLORATION_BONUS,
) -> OptimizationRewardResult:
    """Compute the reward for one configuration evaluation.

    Parameters
    ----------
    objective_score:
        Objective of the evaluated configuration (``None`` on failure).
    previous_best_objective:
        Best objective *before* this evaluation (``None`` if no
        successful evaluation yet).
    region_unexplored:
        ``True`` when neither this configuration nor any immediate
        neighbour had been evaluated before this step.
    """
    if objective_score is None:
        # Configuration failure: zero reward, nothing fabricated.
        return OptimizationRewardResult(
            final_reward=0.0,
            base_reward=None,
            improvement_bonus=None,
            exploration_component=0.0,
            previous_best_objective=previous_best_objective,
            new_best_objective=previous_best_objective,
        )

    base_reward = float(objective_score)

    if previous_best_objective is None:
        improvement_bonus = 0.0
        new_best = base_reward
    else:
        improvement_bonus = max(0.0, base_reward - previous_best_objective)
        new_best = max(previous_best_objective, base_reward)

    exploration_component = (
        float(exploration_bonus) if region_unexplored else 0.0
    )

    final_reward = (
        base_reward
        + improvement_weight * improvement_bonus
        + exploration_component
    )

    return OptimizationRewardResult(
        final_reward=final_reward,
        base_reward=base_reward,
        improvement_bonus=improvement_bonus,
        exploration_component=exploration_component,
        previous_best_objective=previous_best_objective,
        new_best_objective=new_best,
    )