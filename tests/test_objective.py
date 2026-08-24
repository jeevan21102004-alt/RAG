"""Tests for the retrieval objective function.

No API access, internet, or Gemini is required.
"""

import unittest

from src.adaptive_rag.objective import (
    DEFAULT_REFERENCE_LATENCY_MS,
    DEFAULT_WEIGHTS,
    ObjectiveResult,
    calculate_objective,
    normalize_latency,
)


class TestObjectiveCalculation(unittest.TestCase):
    def test_default_objective_value(self) -> None:
        result = calculate_objective(
            retrieval_f1=0.8, context_relevance=0.7, latency_ms=100.0
        )
        # 0.5*0.8 + 0.2*0.7 + 0.3*normalize(100)=0.5
        expected = 0.5 * 0.8 + 0.2 * 0.7 + 0.3 * 0.5
        self.assertIsNotNone(result.objective_score)
        self.assertAlmostEqual(result.objective_score, expected, places=6)

    def test_objective_between_zero_and_one(self) -> None:
        result = calculate_objective(
            retrieval_f1=0.5, context_relevance=0.5, latency_ms=100.0
        )
        self.assertGreaterEqual(result.objective_score, 0.0)
        self.assertLessEqual(result.objective_score, 1.0)

    def test_perfect_scores_give_one(self) -> None:
        result = calculate_objective(
            retrieval_f1=1.0, context_relevance=1.0, latency_ms=0.0
        )
        self.assertAlmostEqual(result.objective_score, 1.0, places=6)

    def test_component_values_are_reported(self) -> None:
        result = calculate_objective(
            retrieval_f1=0.8, context_relevance=0.7, latency_ms=100.0
        )
        self.assertEqual(result.retrieval_component, 0.8)
        self.assertEqual(result.context_component, 0.7)
        self.assertAlmostEqual(result.latency_component, 0.5, places=6)


class TestWeightHandling(unittest.TestCase):
    def test_custom_weights_are_respected(self) -> None:
        weights = {"retrieval_f1": 1.0, "context_relevance": 0.0, "latency": 0.0}
        result = calculate_objective(
            retrieval_f1=0.8, context_relevance=0.2, latency_ms=50.0, weights=weights
        )
        self.assertAlmostEqual(result.objective_score, 0.8, places=6)
        self.assertEqual(result.weights, weights)

    def test_bad_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_objective(
                retrieval_f1=0.8, context_relevance=0.7, latency_ms=10,
                weights={"retrieval_f1": -1.0, "context_relevance": 0.0, "latency": 0.0},
            )

    def test_zero_total_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_objective(
                retrieval_f1=0.8, context_relevance=0.7, latency_ms=10.0,
                weights={"retrieval_f1": 0.0, "context_relevance": 0.0, "latency": 0.0},
            )

    def test_weights_immutable_in_result(self) -> None:
        weights = {"retrieval_f1": 0.5, "context_relevance": 0.2, "latency": 0.3}
        result = calculate_objective(
            retrieval_f1=0.8, context_relevance=0.7, latency_ms=10.0, weights=weights
        )
        weights["retrieval_f1"] = 99.0  # must not affect the frozen result
        self.assertEqual(result.weights["retrieval_f1"], 0.5)


class TestLatencyNormalization(unittest.TestCase):
    def test_zero_latency_scores_one(self) -> None:
        self.assertAlmostEqual(normalize_latency(0.0), 1.0, places=6)

    def test_reference_latency_scores_half(self) -> None:
        self.assertAlmostEqual(
            normalize_latency(DEFAULT_REFERENCE_LATENCY_MS), 0.5, places=6
        )

    def test_latency_above_reference_low_but_positive(self) -> None:
        score = normalize_latency(1000.0)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_missing_latency_returns_none(self) -> None:
        self.assertIsNone(normalize_latency(None))

    def test_negative_latency_clamped_to_zero_score(self) -> None:
        self.assertAlmostEqual(normalize_latency(-5.0), 1.0, places=6)


class TestMissingMetricHandling(unittest.TestCase):
    def test_missing_context_renormalizes_weights(self) -> None:
        result = calculate_objective(
            retrieval_f1=0.8, context_relevance=None, latency_ms=100.0
        )
        # (0.5*0.8 + 0.3*0.5) / 0.8 = 0.6875
        self.assertAlmostEqual(result.objective_score, 0.6875, places=6)
        self.assertIn("context_relevance", result.missing_components)

    def test_missing_all_returns_none_score(self) -> None:
        result = calculate_objective(
            retrieval_f1=None, context_relevance=None, latency_ms=None
        )
        self.assertIsNone(result.objective_score)
        self.assertEqual(
            set(result.missing_components),
            {"retrieval_f1", "context_relevance", "latency"},
        )

    def test_missing_f1_and_latency(self) -> None:
        result = calculate_objective(
            retrieval_f1=None, context_relevance=0.6, latency_ms=None
        )
        self.assertAlmostEqual(result.objective_score, 0.6, places=6)
        self.assertIn("retrieval_f1", result.missing_components)
        self.assertIn("latency", result.missing_components)


class TestDeterministicOutput(unittest.TestCase):
    def test_same_input_same_output(self) -> None:
        a = calculate_objective(
            retrieval_f1=0.7, context_relevance=0.6, latency_ms=80.0
        ).to_dict()
        b = calculate_objective(
            retrieval_f1=0.7, context_relevance=0.6, latency_ms=80.0
        ).to_dict()
        self.assertEqual(a, b)

    def test_returns_objective_result(self) -> None:
        result = calculate_objective(
            retrieval_f1=0.7, context_relevance=0.6, latency_ms=80.0
        )
        self.assertIsInstance(result, ObjectiveResult)
        self.assertEqual(result.weights, DEFAULT_WEIGHTS)
        self.assertIsInstance(result.missing_components, list)


if __name__ == "__main__":
    unittest.main()