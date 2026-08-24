"""Tests for the experiment runner."""

import unittest

from src.adaptive_rag.evaluation_schema import EvaluationCase
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_runner import run_experiment
from src.adaptive_rag.experiment_result import ExperimentResult
from src.adaptive_rag.system_adapter import MockSystemAdapter


def _make_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        ),
        EvaluationCase(
            case_id="c2",
            question="What is 2 + 2?",
            category="retrieval_not_required",
            expected_answer="Four.",
            expected_action="ANSWER",
            relevant_documents=[],
        ),
    ]


class TestExperimentRunner(unittest.TestCase):
    def test_mock_adapter_execution(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)

        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(result.experiment_id, "exp-001")
        self.assertEqual(result.total_cases, 2)
        self.assertEqual(result.successful_cases, 2)
        self.assertEqual(result.failed_cases, 0)

    def test_multiple_evaluation_cases(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-002",
            name="Multi",
            parameters={"top_k": 5, "chunk_size": 300},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)

        self.assertEqual(result.total_cases, 2)
        self.assertEqual(result.successful_cases, 2)

    def test_aggregation(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-003",
            name="Agg",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)

        self.assertIsNotNone(result.average_overall_score)
        self.assertIsNotNone(result.average_answer_score)
        self.assertIsNotNone(result.average_retrieval_score)
        self.assertIsNotNone(result.average_decision_score)
        self.assertIsNotNone(result.average_latency_ms)
        self.assertGreater(result.total_retrieval_attempts, 0)

    def test_diagnostics_integration(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-004",
            name="Diag",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)

        self.assertIn("total_cases", result.diagnostics)
        self.assertIn("successful_cases", result.diagnostics)
        self.assertIn("failure_counts", result.diagnostics)

    def test_latency_handling(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-005",
            name="Latency",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)

        # Mock adapter provides latency, so it should be set
        self.assertIsNotNone(result.average_latency_ms)
        self.assertGreater(result.average_latency_ms, 0)

    def test_empty_cases(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-006",
            name="Empty",
            parameters={"top_k": 3},
        )
        adapter = MockSystemAdapter()
        result = run_experiment(config, [], adapter)

        self.assertEqual(result.total_cases, 0)
        self.assertEqual(result.successful_cases, 0)
        self.assertIsNone(result.average_overall_score)

    def test_result_is_json_serializable(self) -> None:
        import json
        config = ExperimentConfig(
            experiment_id="exp-007",
            name="JSON",
            parameters={"top_k": 3},
        )
        cases = _make_cases()
        adapter = MockSystemAdapter()
        result = run_experiment(config, cases, adapter)
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
