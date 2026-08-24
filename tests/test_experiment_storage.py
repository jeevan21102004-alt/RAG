"""Tests for experiment storage."""

import json
import tempfile
import unittest
from pathlib import Path

from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_result import ExperimentResult
from src.adaptive_rag.experiment_storage import (
    load_experiment_result,
    save_experiment_result,
)


class TestExperimentStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmpdir) / "test_result.json"

    def _make_result(self) -> ExperimentResult:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test Experiment",
            description="A test",
            parameters={"top_k": 3, "chunk_size": 300},
            tags=["test"],
        )
        return ExperimentResult(
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
            diagnostics={"total_cases": 10},
            metadata={"key": "value"},
        )

    def test_save(self) -> None:
        result = self._make_result()
        path = save_experiment_result(result, path=self.tmp_path)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "exp-001")

    def test_load(self) -> None:
        result = self._make_result()
        save_experiment_result(result, path=self.tmp_path)
        loaded = load_experiment_result(self.tmp_path)
        self.assertEqual(loaded.experiment_id, "exp-001")
        self.assertEqual(loaded.total_cases, 10)
        self.assertEqual(loaded.successful_cases, 8)
        self.assertEqual(loaded.average_overall_score, 0.80)

    def test_round_trip_equality(self) -> None:
        result = self._make_result()
        save_experiment_result(result, path=self.tmp_path)
        loaded = load_experiment_result(self.tmp_path)

        self.assertEqual(loaded.experiment_id, result.experiment_id)
        self.assertEqual(loaded.total_cases, result.total_cases)
        self.assertEqual(loaded.successful_cases, result.successful_cases)
        self.assertEqual(loaded.failed_cases, result.failed_cases)
        self.assertEqual(loaded.average_answer_score, result.average_answer_score)
        self.assertEqual(loaded.average_retrieval_score, result.average_retrieval_score)
        self.assertEqual(loaded.average_decision_score, result.average_decision_score)
        self.assertEqual(loaded.average_overall_score, result.average_overall_score)
        self.assertEqual(loaded.average_latency_ms, result.average_latency_ms)
        self.assertEqual(loaded.total_retrieval_attempts, result.total_retrieval_attempts)
        self.assertEqual(loaded.diagnostics, result.diagnostics)
        self.assertEqual(loaded.metadata, result.metadata)
        self.assertEqual(loaded.config.experiment_id, result.config.experiment_id)
        self.assertEqual(loaded.config.parameters, result.config.parameters)
        self.assertEqual(loaded.config.tags, result.config.tags)

    def test_save_default_path(self) -> None:
        result = self._make_result()
        # Use a temp directory as the results dir
        import src.adaptive_rag.experiment_storage as storage
        original = storage.RESULTS_DIR
        storage.RESULTS_DIR = Path(self.tmpdir) / "results"
        try:
            path = save_experiment_result(result)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "exp-001.json")
        finally:
            storage.RESULTS_DIR = original


if __name__ == "__main__":
    unittest.main()
