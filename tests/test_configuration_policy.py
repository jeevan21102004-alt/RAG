"""Tests for the configuration RL policy and action masking (Phase 4)."""

import unittest

import torch

from src.adaptive_rag.configuration_policy import (
    ConfigurationPolicy,
    mask_invalid_actions,
)
from src.adaptive_rag.optimization_state import OPTIMIZATION_STATE_DIM
from src.adaptive_rag.search_space import SearchSpace


def _policy(seed=42, hidden=64):
    return ConfigurationPolicy(
        n_actions=175,
        state_dim=OPTIMIZATION_STATE_DIM,
        hidden_dim=hidden,
        seed=seed,
    )


class TestPolicyOutput(unittest.TestCase):
    def test_output_shape(self) -> None:
        p = _policy()
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        logits = p(state)
        self.assertEqual(logits.shape, (1, 175))

    def test_probabilities_sum_to_one(self) -> None:
        p = _policy()
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        probs = p.probabilities(state)
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=5)
        self.assertTrue(torch.all(probs >= 0).item())

    def test_deterministic_seed(self) -> None:
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        probs_a = _policy(seed=42)(state)
        probs_b = _policy(seed=42)(state)
        self.assertTrue(torch.allclose(probs_a, probs_b))


class TestActionMasking(unittest.TestCase):
    def test_masked_probabilities_are_zero(self) -> None:
        p = _policy()
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        valid = torch.ones(1, 175, dtype=torch.bool)
        # Mask the first 10 actions.
        valid[0, :10] = False
        probs = p.probabilities(state, valid)
        self.assertTrue(torch.all(probs[0, :10] == 0).item())
        self.assertAlmostEqual(float(probs[0, 10:].sum().item()), 1.0, places=5)

    def test_at_least_one_valid_remains(self) -> None:
        p = _policy()
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        valid = torch.ones(1, 175, dtype=torch.bool)
        # Mask 174 — one remains.
        valid[0, :174] = False
        probs = p.probabilities(state, valid)
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=5)
        self.assertEqual(int(torch.argmax(probs).item()), 174)

    def test_mask_last_action_keeps_first(self) -> None:
        p = _policy()
        state = torch.zeros(1, OPTIMIZATION_STATE_DIM)
        valid = torch.ones(1, 175, dtype=torch.bool)
        valid[0, 174] = False
        probs = p.probabilities(state, valid)
        self.assertEqual(float(probs[0, 174].item()), 0.0)


if __name__ == "__main__":
    unittest.main()