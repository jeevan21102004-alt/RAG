"""REINFORCE training for RL configuration selection (Phase 4).

Direct PyTorch policy-gradient (REINFORCE) with a masked action space.

For each episode:

1. ``reset()`` the environment.
2. Build the state vector.
3. Mask already-evaluated actions.
4. Sample an action from the policy.
5. ``step()`` the environment (runs ONE local retrieval experiment).
6. Record log-probability and reward.
7. Repeat until the episode is done (budget exhausted / space exhausted).
8. Discounted returns -> policy-gradient loss -> Adam update.

Training is entirely local: the retrieval benchmark runs generation-
disabled, so **no Gemini calls and no internet access** occur.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .configuration_policy import ConfigurationPolicy
from .evaluation_schema import EvaluationCase
from .objective import ObjectiveConfig
from .optimization_state import OPTIMIZATION_STATE_DIM, optimization_state_to_vector
from .rl_optimization_environment import RetrievalOptimizationEnvironment
from .search_space import SearchSpace
from .system_adapter import SystemAdapter

DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_GAMMA: float = 0.95
DEFAULT_SEED: int = 42


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute Monte-Carlo discounted returns from an episode's rewards."""
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


class ReinforceTrainer:
    """REINFORCE trainer for the configuration-selection policy."""

    def __init__(
        self,
        search_space: SearchSpace,
        cases: list[EvaluationCase],
        adapter: SystemAdapter,
        episodes: int = 100,
        budget: int = 8,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        gamma: float = DEFAULT_GAMMA,
        seed: int = DEFAULT_SEED,
        hidden_dim: int = 64,
        objective_config: ObjectiveConfig | None = None,
        device: str | None = None,
    ) -> None:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        if budget < 1:
            raise ValueError("budget must be >= 1")

        set_seed(seed)
        self.search_space = search_space
        self.cases = cases
        self.adapter = adapter
        self.episodes = int(episodes)
        self.budget = int(budget)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        self.objective_config = objective_config or ObjectiveConfig()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        n_actions = search_space.count_combinations()
        self.n_actions = n_actions
        self.policy = ConfigurationPolicy(
            n_actions=n_actions,
            state_dim=OPTIMIZATION_STATE_DIM,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        self.episode_rewards: list[float] = []
        self.episode_objectives: list[float | None] = []
        self.episode_losses: list[float] = []
        self.best_episode_objective: float | None = None

    def _run_episode(self) -> dict[str, Any]:
        env = RetrievalOptimizationEnvironment(
            search_space=self.search_space,
            cases=self.cases,
            adapter=self.adapter,
            budget=self.budget,
            objective_config=self.objective_config,
        )
        state = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False

        while not done:
            state_tensor = torch.tensor(
                optimization_state_to_vector(state),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            valid_mask = torch.tensor(
                env.valid_action_mask(),
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)

            # Compute the masked distribution WITH gradients (act() uses
            # no_grad, so we derive the sampled action's log-prob here).
            probs = self.policy.probabilities(state_tensor, valid_mask)
            if torch.all(probs == 0):
                raise RuntimeError("All actions masked — no valid action remains.")
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
            log_prob = dist.log_prob(torch.tensor([action], device=self.device))[0]

            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state

        returns = discounted_returns(rewards, self.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = float(loss.item())

        objectives = [
            record["objective"] for record in env._order if record["objective"] is not None
        ]
        episode_best = max(objectives) if objectives else None

        return {
            "loss": loss_value,
            "total_reward": float(sum(rewards)),
            "evaluations": env.evaluations_used,
            "best_objective": episode_best,
            "rewards": rewards,
        }

    def train(self) -> dict[str, Any]:
        """Run *episodes* REINFORCE episodes; return a training summary."""
        for episode in range(1, self.episodes + 1):
            summary = self._run_episode()
            self.episode_rewards.append(summary["total_reward"])
            self.episode_losses.append(summary["loss"])
            self.episode_objectives.append(summary["best_objective"])
            if summary["best_objective"] is not None and (
                self.best_episode_objective is None
                or summary["best_objective"] > self.best_episode_objective
            ):
                self.best_episode_objective = summary["best_objective"]

        finite = all(float(l) == float(l) and abs(float(l)) != float("inf") for l in self.episode_losses)
        return {
            "episodes": self.episodes,
            "budget": self.budget,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "losses": self.episode_losses,
            "loss_finite": finite,
            "episode_rewards": self.episode_rewards,
            "episode_objectives": self.episode_objectives,
            "best_episode_objective": self.best_episode_objective,
            "final_loss": self.episode_losses[-1] if self.episode_losses else None,
            "n_actions": self.n_actions,
            "state_dim": OPTIMIZATION_STATE_DIM,
            "hidden_dim": self.hidden_dim,
            "device": str(self.device),
            "api_calls": 0,
        }