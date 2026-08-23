from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import ANSWER, SEARCH
from .reward import calculate_reward
from .state import question_to_state


ACTION_ANSWER = 0
ACTION_SEARCH = 1

ACTION_TO_LABEL = {
    ACTION_ANSWER: ANSWER,
    ACTION_SEARCH: SEARCH,
}


@dataclass
class RetrievalDecisionEnvironment:
    expected_action: str
    question: str = ""
    state: list[float] = field(default_factory=list)
    done: bool = False

    def reset(self, question: str) -> list[float]:
        self.question = question
        self.state = question_to_state(question)
        self.done = False
        return self.state

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() first.")

        actual_action = ACTION_TO_LABEL[action]
        reward_result = calculate_reward(self.expected_action, actual_action)

        self.done = True

        metadata = {
            "expected_action": self.expected_action,
            "actual_action": actual_action,
            "reward_reason": reward_result.reason,
        }

        return self.state, float(reward_result.reward), self.done, metadata