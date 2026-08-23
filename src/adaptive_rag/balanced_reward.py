from __future__ import annotations

from dataclasses import dataclass

from .agent import ANSWER, SEARCH


REWARD_CORRECT = 1
REWARD_UNNECESSARY_SEARCH = -1
REWARD_MISSED_SEARCH = -1


@dataclass(frozen=True)
class BalancedRewardResult:
    expected_action: str
    actual_action: str
    reward: int
    reason: str


def calculate_balanced_reward(expected_action: str, actual_action: str) -> BalancedRewardResult:
    """Calculate a balanced deterministic reward.

    Balanced scheme:
      Correct SEARCH: +1
      Correct ANSWER: +1
      Unnecessary SEARCH: -1
      Missed SEARCH: -1

    This is a pure function and does NOT call any LLM or API.
    """
    if expected_action == SEARCH and actual_action == SEARCH:
        return BalancedRewardResult(
            expected_action=expected_action,
            actual_action=actual_action,
            reward=REWARD_CORRECT,
            reason="The agent correctly identified that retrieval was required.",
        )

    if expected_action == ANSWER and actual_action == ANSWER:
        return BalancedRewardResult(
            expected_action=expected_action,
            actual_action=actual_action,
            reward=REWARD_CORRECT,
            reason="The agent correctly identified that retrieval was not required.",
        )

    if expected_action == ANSWER and actual_action == SEARCH:
        return BalancedRewardResult(
            expected_action=expected_action,
            actual_action=actual_action,
            reward=REWARD_UNNECESSARY_SEARCH,
            reason="The agent performed an unnecessary search when retrieval was not required.",
        )

    if expected_action == SEARCH and actual_action == ANSWER:
        return BalancedRewardResult(
            expected_action=expected_action,
            actual_action=actual_action,
            reward=REWARD_MISSED_SEARCH,
            reason="The agent missed the search and answered without retrieval when retrieval was required.",
        )

    return BalancedRewardResult(
        expected_action=expected_action,
        actual_action=actual_action,
        reward=0,
        reason="Unknown action pair; no reward assigned.",
    )