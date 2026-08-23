import unittest

from src.adaptive_rag.agent import ANSWER, SEARCH
from src.adaptive_rag.reward import (
    REWARD_CORRECT,
    REWARD_MISSED_SEARCH,
    REWARD_UNNECESSARY_SEARCH,
    calculate_reward,
)


class TestRewardCalculation(unittest.TestCase):
    def test_correct_search(self) -> None:
        result = calculate_reward(SEARCH, SEARCH)
        self.assertEqual(result.reward, REWARD_CORRECT)
        self.assertEqual(result.expected_action, SEARCH)
        self.assertEqual(result.actual_action, SEARCH)

    def test_correct_answer(self) -> None:
        result = calculate_reward(ANSWER, ANSWER)
        self.assertEqual(result.reward, REWARD_CORRECT)
        self.assertEqual(result.expected_action, ANSWER)
        self.assertEqual(result.actual_action, ANSWER)

    def test_unnecessary_search(self) -> None:
        result = calculate_reward(ANSWER, SEARCH)
        self.assertEqual(result.reward, REWARD_UNNECESSARY_SEARCH)
        self.assertEqual(result.expected_action, ANSWER)
        self.assertEqual(result.actual_action, SEARCH)

    def test_missed_search(self) -> None:
        result = calculate_reward(SEARCH, ANSWER)
        self.assertEqual(result.reward, REWARD_MISSED_SEARCH)
        self.assertEqual(result.expected_action, SEARCH)
        self.assertEqual(result.actual_action, ANSWER)


if __name__ == "__main__":
    unittest.main()