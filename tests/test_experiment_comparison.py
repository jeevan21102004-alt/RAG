"""Tests for experiment comparison."""

import unittest

from src.adaptive_rag.experiment_comparison import compare_experiments
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_result import ExperimentResult


def _make_result(
    experiment_id: str,
    overall: float | None = None,
    answer: float | None = None,
    retrieval: float | None = None,
    latency: float | None = None,
) -> ExperimentResult:
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_id,
        parameters={"top_k": 3},
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        total_cases=10,
        average_overall_score=overall,
        average_answer_score=answer,
        average_retrieval_score=retrieval,
        average_latency_ms=latency,
    )


class TestExperimentComparison(unittest.TestCase):
    def test_ranking(self) -> None:
        results = [
            _make_result("a", overall=0.5),
            _make_result("b", overall=0.8),
            _make_result("c", overall=0.3),
        ]
        comparison = compare_experiments(results)
        self.assertEqual(comparison.ranking, ["b", "a", "c"])

    def test_best_overall(self) -> None:
        results = [
            _make_result("a", overall=0.5),
            _make_result("b", overall=0.8),
            _make_result("c", overall=0.3),
        ]
        comparison = compare_experiments(results)
        self.assertEqual(comparison.best_by_overall_score, "b")

    def test_best_answer(self) -> None:
        results = [
            _make_result("a", answer=0.5),
            _make_result("b", answer=0.9),
            _make_result("c", answer=0.3),
        ]
        comparison = compare_experiments(results)
        self.assertEqual(comparison.best_by_answer_score, "b")

    def test_best_retrieval(self) -> None:
        results = [
            _make_result("a", retrieval=0.5),
            _make_result("b", retrieval=0.9),
            _make_result("c", retrieval=0.3),
        ]
        comparison = compare_experiments(results)
        self.assertEqual(comparison.best_by_retrieval_score, "b")

    def test_fastest_experiment(self) -> None:
        results = [
            _make_result("a", latency=500.0),
            _make_result("b", latency=200.0),
            _make_result("c", latency=800.0),
        ]
        comparison = compare_experiments(results)
        self.assertEqual(comparison.fastest_experiment, "b")

    def test_missing_metric_handling(self) -> None:
        results = [
            _make_result("a", overall=0.5, answer=None),
            _make_result("b", overall=None, answer=0.9),
        ]
        comparison = compare_experiments(results)
        # b has no overall score, so it should be ranked last
        self.assertEqual(comparison.ranking, ["a", "b"])
        self.assertEqual(comparison.best_by_overall_score, "a")
        self.assertEqual(comparison.best_by_answer_score, "b")

    def test_empty_results(self) -> None:
        comparison = compare_experiments([])
        self.assertEqual(comparison.experiments, [])
        self.assertIsNone(comparison.best_by_overall_score)
        self.assertEqual(comparison.ranking, [])

    def test_to_dict(self) -> None:
        results = [
            _make_result("a", overall=0.5),
            _make_result("b", overall=0.8),
        ]
        comparison = compare_experiments(results)
        d = comparison.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("experiments", d)
        self.assertIn("best_by_overall_score", d)
        self.assertIn("ranking", d)


if __name__ == "__main__":
    unittest.main()
