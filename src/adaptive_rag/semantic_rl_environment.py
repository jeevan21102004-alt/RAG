from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import ANSWER, SEARCH
from .balanced_reward import calculate_balanced_reward
from .reward import calculate_reward
from .semantic_state import question_to_embedding


ACTION_ANSWER = 0
ACTION_SEARCH = 1

ACTION_TO_LABEL = {
    ACTION_ANSWER: ANSWER,
    ACTION_SEARCH: SEARCH,
}


@dataclass
class SemanticRetrievalDecisionEnvironment:
    """A single-decision RL environment using semantic embeddings as state.

    The reward function is configurable so experiments can compare different
    reward schemes without modifying the original environment.
    """

    expected_action: str
    reward_function: Callable = calculate_reward
    question: str = ""
    state: list[float] = field(default_factory=list)
    done: bool = False

    def reset(self, question: str) -> list[float]:
        """Start a new episode with the given question."""
        self.question = question
        self.state = question_to_embedding(question)
        self.done = False
        return self.state

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Take an action and return (next_state, reward, done, metadata).

        Since this is a single-decision environment, the episode ends after
        one action.
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() first.")

        actual_action = ACTION_TO_LABEL[action]
        reward_result = self.reward_function(self.expected_action, actual_action)

        self.done = True

        metadata = {
            "expected_action": self.expected_action,
            "actual_action": actual_action,
            "reward_reason": reward_result.reason,
        }

        return self.state, float(reward_result.reward), self.done, metadata