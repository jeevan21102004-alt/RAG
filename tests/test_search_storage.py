"""Tests for search result JSON storage.

No API access, internet, or Gemini is required.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.adaptive_rag.search_result import SearchRankingEntry, SearchResult
from src.adaptive_rag.search_storage import load_search_result, save_search_result


def _make_result() -> SearchResult:
    ranking = [
        SearchRankingEntry(
            experiment_id="grid-1",
            parameters={"chunk_size": 100, "chunk_overlap": 20, "top_k": 5},
            objective_score=0.9,
            retrieval_f1=0.8,
            context_relevance=0.7,
            latency_ms=50.0,
            status="SUCCESS",
        ),
        SearchRankingEntry(
            experiment_id="grid-2",
            parameters={"chunk_size": 300, "chunk_overlap": 50, "top_k": 3},
            objective_score=None,
            retrieval_f1=None,
            context_relevance=None,
            latency_ms=None,
            status="CONFIG_ERROR: mock failure",
        ),
    ]
    return SearchResult(
        search_id="test-search",
        total_configurations=2,
        completed_configurations=1,
        failed_configurations=1,
        ranking=ranking,
        best_configuration=ranking[0],
        best_objective_score=0.9,
        objective_definition={
            "weights": {"retrieval_f1": 0.5, "context_relevance": 0.2, "latency": 0.3},
            "reference_latency_ms": 100.0,
        },
        metadata={"dataset": "enterprise", "questions_per_config": 2},
    )


class TestSaveSearchResult(unittest.TestCase):
    def test_save_creates_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_search_result(_make_result(), Path(tmp) / "out.json")
            self.assertTrue(path.exists())

    def test_saved_content_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_search_result(_make_result(), Path(tmp) / "out.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["search_id"], "test-search")

    def test_save_creates_missing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "out.json"
            path = save_search_result(_make_result(), nested)
            self.assertTrue(path.exists())


class TestLoadSearchResult(unittest.TestCase):
    def test_load_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = _make_result()
            path = save_search_result(original, Path(tmp) / "out.json")
            loaded = load_search_result(path)

        self.assertEqual(loaded.search_id, original.search_id)
        self.assertEqual(loaded.total_configurations, original.total_configurations)
        self.assertEqual(
            loaded.completed_configurations, original.completed_configurations
        )
        self.assertEqual(loaded.failed_configurations, original.failed_configurations)
        self.assertEqual(loaded.best_objective_score, original.best_objective_score)
        self.assertIsNotNone(loaded.best_configuration)
        self.assertEqual(
            loaded.best_configuration.experiment_id,
            original.best_configuration.experiment_id,
        )
        self.assertEqual(loaded.objective_definition, original.objective_definition)
        self.assertEqual(loaded.metadata, original.metadata)


class TestJsonSerialization(unittest.TestCase):
    def test_round_trip_preserves_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = _make_result()
            path = save_search_result(original, Path(tmp) / "out.json")
            loaded = load_search_result(path)

        self.assertEqual(len(loaded.ranking), 2)
        first = loaded.ranking[0]
        self.assertEqual(first.experiment_id, "grid-1")
        self.assertAlmostEqual(first.objective_score, 0.9)
        self.assertAlmostEqual(first.retrieval_f1, 0.8)
        self.assertEqual(first.parameters["chunk_size"], 100)
        self.assertEqual(first.status, "SUCCESS")
        second = loaded.ranking[1]
        self.assertIsNone(second.objective_score)
        self.assertTrue(second.status.startswith("CONFIG_ERROR"))

    def test_to_dict_is_json_serializable(self) -> None:
        payload = json.dumps(_make_result().to_dict())
        self.assertIn("test-search", payload)


if __name__ == "__main__":
    unittest.main()