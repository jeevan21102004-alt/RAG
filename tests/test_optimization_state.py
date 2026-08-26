"""Tests for the optimization state (Phase 4)."""

import unittest

from src.adaptive_rag.optimization_state import (
    OPTIMIZATION_STATE_DIM,
    OptimizationState,
    optimization_state_to_vector,
)


class TestStateVector(unittest.TestCase):
    def test_default_state_dimension(self) -> None:
        self.assertEqual(OPTIMIZATION_STATE_DIM, 15)

    def test_default_state_vector(self) -> None:
        state = OptimizationState()
        vec = optimization_state_to_vector(state)
        self.assertEqual(len(vec), 15)
        self.assertEqual(vec[0], 0.0)   # best objective
        self.assertEqual(vec[10], 1.0)  # unexplored fraction
        self.assertEqual(vec[11], 0.0)  # last improved

    def test_populated_state_vector(self) -> None:
        state = OptimizationState(
            best_objective=0.8,
            best_f1=0.9,
            best_context_relevance=0.85,
            best_latency_score=0.7,
            budget_consumed_fraction=0.5,
            space_evaluated_fraction=0.25,
            successful_fraction=0.5,
            failed_fraction=0.0,
            average_objective=0.6,
            last_improvement=0.1,
            unexplored_fraction=0.75,
            last_action_improved=1.0,
            prev_chunk_size=0.5,
            prev_chunk_overlap=0.5,
            prev_top_k=0.5,
        )
        vec = optimization_state_to_vector(state)
        self.assertEqual(len(vec), 15)
        self.assertEqual(vec[0], 0.8)
        self.assertEqual(vec[11], 1.0)

    def test_deterministic(self) -> None:
        a = optimization_state_to_vector(OptimizationState(best_objective=0.5))
        b = optimization_state_to_vector(OptimizationState(best_objective=0.5))
        self.assertEqual(a, b)

    def test_index_normalization(self) -> None:
        from src.adaptive_rag.optimization_state import _normalize_index

        self.assertEqual(_normalize_index(100, [100, 200, 300]), 0.0)
        self.assertEqual(_normalize_index(200, [100, 200, 300]), 0.5)
        self.assertEqual(_normalize_index(300, [100, 200, 300]), 1.0)
        self.assertEqual(_normalize_index(None, [100]), 0.0)
        self.assertEqual(_normalize_index(999, [100, 200]), 0.0)


if __name__ == "__main__":
    unittest.main()