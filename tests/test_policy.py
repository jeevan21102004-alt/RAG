import unittest

import torch

from src.adaptive_rag.policy import RetrievalPolicy
from src.adaptive_rag.state import question_to_state


class TestRetrievalPolicy(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.policy = RetrievalPolicy()

    def test_model_accepts_5_feature_state(self) -> None:
        state = torch.tensor(question_to_state("What is machine learning?"), dtype=torch.float32)
        probs = self.policy.action_probs(state)
        self.assertEqual(probs.shape, torch.Size([2]))

    def test_output_has_2_action_probabilities(self) -> None:
        state = torch.tensor(question_to_state("What is machine learning?"), dtype=torch.float32)
        probs = self.policy.action_probs(state)
        self.assertEqual(probs.shape[0], 2)

    def test_probabilities_sum_to_approximately_1(self) -> None:
        state = torch.tensor(question_to_state("What is machine learning?"), dtype=torch.float32)
        probs = self.policy.action_probs(state)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_action_is_either_0_or_1(self) -> None:
        state = torch.tensor(question_to_state("What is machine learning?"), dtype=torch.float32)
        action, _ = self.policy.sample_action(state)
        self.assertIn(action, (0, 1))

    def test_forward_pass_on_cpu(self) -> None:
        state = torch.tensor(question_to_state("What is machine learning?"), dtype=torch.float32)
        probs = self.policy.action_probs(state)
        self.assertEqual(probs.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()