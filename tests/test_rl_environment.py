import unittest

from src.adaptive_rag.agent import ANSWER, SEARCH
from src.adaptive_rag.rl_environment import (
    ACTION_ANSWER,
    ACTION_SEARCH,
    ACTION_TO_LABEL,
    RetrievalDecisionEnvironment,
)
from src.adaptive_rag.state import question_to_state, state_feature_names


class TestStateRepresentation(unittest.TestCase):
    def test_same_question_same_state(self) -> None:
        question = "What is supervised learning?"
        self.assertEqual(question_to_state(question), question_to_state(question))

    def test_state_has_expected_number_of_features(self) -> None:
        state = question_to_state("What is machine learning?")
        self.assertEqual(len(state), len(state_feature_names()))


class TestActionMapping(unittest.TestCase):
    def test_action_0_is_answer(self) -> None:
        self.assertEqual(ACTION_ANSWER, 0)
        self.assertEqual(ACTION_TO_LABEL[ACTION_ANSWER], ANSWER)

    def test_action_1_is_search(self) -> None:
        self.assertEqual(ACTION_SEARCH, 1)
        self.assertEqual(ACTION_TO_LABEL[ACTION_SEARCH], SEARCH)


class TestEnvironmentRewards(unittest.TestCase):
    def test_correct_search_receives_plus_2(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=SEARCH)
        env.reset("What is machine learning?")
        _, reward, done, _ = env.step(ACTION_SEARCH)
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)

    def test_correct_answer_receives_plus_2(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=ANSWER)
        env.reset("What is 2 + 2?")
        _, reward, done, _ = env.step(ACTION_ANSWER)
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)

    def test_unnecessary_search_receives_minus_1(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=ANSWER)
        env.reset("What is 2 + 2?")
        _, reward, done, _ = env.step(ACTION_SEARCH)
        self.assertEqual(reward, -1.0)
        self.assertTrue(done)

    def test_missed_search_receives_minus_3(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=SEARCH)
        env.reset("What is machine learning?")
        _, reward, done, _ = env.step(ACTION_ANSWER)
        self.assertEqual(reward, -3.0)
        self.assertTrue(done)


class TestEpisodeLifecycle(unittest.TestCase):
    def test_episode_ends_after_one_action(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=SEARCH)
        env.reset("What is machine learning?")
        _, _, done, _ = env.step(ACTION_SEARCH)
        self.assertTrue(done)

    def test_step_after_done_raises(self) -> None:
        env = RetrievalDecisionEnvironment(expected_action=SEARCH)
        env.reset("What is machine learning?")
        env.step(ACTION_SEARCH)
        with self.assertRaises(RuntimeError):
            env.step(ACTION_SEARCH)


if __name__ == "__main__":
    unittest.main()