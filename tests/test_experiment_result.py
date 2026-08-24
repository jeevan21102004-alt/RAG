"""Tests for ExperimentResult."""

import json
import unittest

from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_result import ExperimentResult


class TestExperimentResultCreation(unittest.TestCase):
    def test_result_creation(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
            total_cases=10,
            successful_cases=8,
            failed_cases=2,
            average_answer_score=0.85,
            average_retrieval_score=0.70,
            average_decision_score=0.90,
            average_overall_score=0.80,
            average_latency_ms=150.0,
            total_retrieval_attempts=12,
        )
        self.assertEqual(result.experiment_id, "exp-001")
        self.assertEqual(result.total_cases, 10)
        self.assertEqual(result.successful_cases, 8)
        self.assertEqual(result.failed_cases, 2)
        self.assertEqual(result.average_answer_score, 0.85)
        self.assertEqual(result.average_overall_score, 0.80)
        self.assertEqual(result.total_retrieval_attempts, 12)

    def test_result_defaults(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
        )
        self.assertEqual(result.total_cases, 0)
        self.assertEqual(result.successful_cases, 0)
        self.assertEqual(result.failed_cases, 0)
        self.assertIsNone(result.average_answer_score)
        self.assertIsNone(result.average_retrieval_score)
        self.assertIsNone(result.average_decision_score)
        self.assertIsNone(result.average_overall_score)
        self.assertIsNone(result.average_latency_ms)
        self.assertEqual(result.total_retrieval_attempts, 0)
        self.assertEqual(result.diagnostics, {})
        self.assertEqual(result.metadata, {})


class TestExperimentResultSerialization(unittest.TestCase):
    def test_to_dict(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
            total_cases=5,
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["experiment_id"], "exp-001")
        self.assertEqual(d["total_cases"], 5)
        self.assertIn("config", d)
        self.assertEqual(d["config"]["experiment_id"], "exp-001")

    def test_to_json(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
            total_cases=5,
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["experiment_id"], "exp-001")
        self.assertEqual(data["total_cases"], 5)

    def test_json_serializable(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
            tags=["test"],
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
            total_cases=5,
            average_overall_score=0.85,
            diagnostics={"total_cases": 5},
            metadata={"key": "value"},
        )
        json.dumps(result.to_dict())


class TestExperimentResultImmutability(unittest.TestCase):
    def test_frozen_dataclass(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
        )
        result = ExperimentResult(
            experiment_id="exp-001",
            config=config,
        )
        with self.assertRaises(Exception):
            result.total_cases = 10  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
