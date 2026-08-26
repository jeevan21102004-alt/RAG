"""Tests for the optimization reward (Phase 4)."""

import unittest

from src.adaptive_rag.optimization_reward import calculate_optimization_reward


class TestOptimizationReward(unittest.TestCase):
    def test_normal_reward(self) -> None:
        r = calculate_optimization_reward(
            objective_score=0.7,
            previous_best_objective=0.5,
            region_unexplored=False,
        )
        # final = 0.7 + 0.5*(0.7-0.5) = 0.7 + 0.1 = 0.8
        self.assertAlmostEqual(r.final_reward, 0.8)
        self.assertAlmostEqual(r.base_reward, 0.7)
        self.assertAlmostEqual(r.improvement_bonus, 0.2)
        self.assertEqual(r.exploration_component, 0.0)

    def test_improvement_bonus(self) -> None:
        r = calculate_optimization_reward(
            objective_score=0.9,
            previous_best_objective=0.6,
            region_unexplored=False,
        )
        self.assertAlmostEqual(r.improvement_bonus, 0.3)
        self.assertAlmostEqual(r.final_reward, 0.9 + 0.5 * 0.3)

    def test_no_improvement(self) -> None:
        r = calculate_optimization_reward(
            objective_score=0.4,
            previous_best_objective=0.6,
            region_unexplored=False,
        )
        self.assertEqual(r.improvement_bonus, 0.0)
        self.assertAlmostEqual(r.final_reward, 0.4)

    def test_first_success_has_no_improvement_bonus(self) -> None:
        r = calculate_optimization_reward(
            objective_score=0.5,
            previous_best_objective=None,
            region_unexplored=False,
        )
        self.assertAlmostEqual(r.final_reward, 0.5)
        self.assertEqual(r.improvement_bonus, 0.0)

    def test_exploration_bonus(self) -> None:
        r = calculate_optimization_reward(
            objective_score=0.5,
            previous_best_objective=0.5,
            region_unexplored=True,
        )
        self.assertAlmostEqual(r.exploration_component, 0.05)
        self.assertAlmostEqual(r.final_reward, 0.5 + 0.05)

    def test_failed_evaluation_zero_reward(self) -> None:
        r = calculate_optimization_reward(
            objective_score=None,
            previous_best_objective=0.6,
            region_unexplored=False,
        )
        self.assertEqual(r.final_reward, 0.0)
        self.assertIsNone(r.base_reward)
        self.assertIsNone(r.improvement_bonus)
        self.assertEqual(r.new_best_objective, 0.6)  # unchanged

    def test_deterministic(self) -> None:
        a = calculate_optimization_reward(0.7, 0.5, False)
        b = calculate_optimization_reward(0.7, 0.5, False)
        self.assertEqual(a.final_reward, b.final_reward)

    def test_reward_range(self) -> None:
        r = calculate_optimization_reward(1.0, 0.0, False)
        self.assertLessEqual(r.final_reward, 1.0 + 0.5 * 1.0)


if __name__ == "__main__":
    unittest.main()