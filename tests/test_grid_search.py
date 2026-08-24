"""Tests for the deterministic grid search engine.

These tests run against a deterministic mock adapter so no Gemini or
internet access is involved.
"""

import unittest

from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.grid_search import (
    GridSearchEngine,
    SearchSystemError,
    load_search_cases,
    rank_entries,
)
from src.adaptive_rag.objective import ObjectiveConfig
from src.adaptive_rag.search_result import SearchRankingEntry
from src.adaptive_rag.search_space import SearchSpace


class DeterministicMockAdapter:
    """Adapter whose retrieval quality varies deterministically with config."""

    def run(
        self,
        case: EvaluationCase,
        config: ExperimentConfig,
    ) -> SystemResponse:
        top_k = config.parameters["top_k"]
        chunk_size = config.parameters["chunk_size"]
        retrieved = list(case.relevant_documents)
        if chunk_size >= 200:
            retrieved.append("distractor.md")
        retrieved = retrieved[:top_k]

        latency_ms = float(top_k * 10 + chunk_size / 10)
        context = " ".join(retrieved) if retrieved else None

        return SystemResponse(
            answer=None,
            action="SEARCH",
            retrieved_documents=retrieved,
            retrieved_context=context,
            latency_ms=latency_ms,
            retrieval_attempts=1,
            metadata={
                "adapter": "deterministic_mock",
                "config_chunk_size": chunk_size,
                "config_chunk_overlap": config.parameters["chunk_overlap"],
                "config_top_k": top_k,
                "generation_latency_ms": None,
            },
        )


class FailingMockAdapter(DeterministicMockAdapter):
    """Adapter that raises for a specific top_k."""

    def __init__(self, fail_top_k: int = 5) -> None:
        self.fail_top_k = fail_top_k

    def run(self, case: EvaluationCase, config: ExperimentConfig) -> SystemResponse:
        if config.parameters.get("top_k") == self.fail_top_k:
            raise RuntimeError("mock adapter failure for this configuration")
        return super().run(case, config)


class AlwaysFailAdapter:
    def run(self, case: EvaluationCase, config: ExperimentConfig) -> SystemResponse:
        raise RuntimeError("boom")


def _make_engine(adapter=None, dataset="enterprise", max_questions=2):
    return GridSearchEngine(
        SearchSpace.pilot(),
        dataset=dataset,
        max_questions=max_questions,
        adapter=adapter if adapter is not None else DeterministicMockAdapter(),
        objective_config=ObjectiveConfig(),
        search_id="test-search",
    )


class TestAllConfigurationsExecuted(unittest.TestCase):
    def test_eight_configurations_evaluated(self) -> None:
        result = _make_engine().run()
        self.assertEqual(result.total_configurations, 8)
        self.assertEqual(len(result.ranking), 8)
        self.assertEqual(result.completed_configurations, 8)
        self.assertEqual(result.failed_configurations, 0)

    def test_every_ranking_entry_has_metadata(self) -> None:
        result = _make_engine().run()
        for entry in result.ranking:
            self.assertTrue(entry.experiment_id)
            self.assertIn("chunk_size", entry.parameters)
            self.assertIn("chunk_overlap", entry.parameters)
            self.assertIn("top_k", entry.parameters)


class TestRanking(unittest.TestCase):
    def test_best_is_first_in_ranking(self) -> None:
        result = _make_engine().run()
        self.assertIsNotNone(result.best_configuration)
        self.assertEqual(
            result.ranking[0].experiment_id,
            result.best_configuration.experiment_id,
        )
        self.assertEqual(
            result.best_configuration.objective_score,
            result.best_objective_score,
        )

    def test_ranking_is_sorted_descending(self) -> None:
        result = _make_engine().run()
        scores = [
            e.objective_score for e in result.ranking
            if e.objective_score is not None
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestBestConfiguration(unittest.TestCase):
    def test_best_exists_when_all_succeed(self) -> None:
        result = _make_engine().run()
        self.assertIsNotNone(result.best_configuration)
        self.assertIn("chunk_size", result.best_configuration.parameters)

    def test_best_has_objective_score(self) -> None:
        result = _make_engine().run()
        self.assertIsNotNone(result.best_configuration.objective_score)


class TestFailureHandling(unittest.TestCase):
    def test_single_config_failure_continues_search(self) -> None:
        adapter = FailingMockAdapter(fail_top_k=5)
        result = _make_engine(adapter=adapter).run()
        self.assertEqual(result.failed_configurations, 4)
        self.assertEqual(result.completed_configurations, 4)
        self.assertEqual(result.total_configurations, 8)
        failed_ids = [
            e.experiment_id for e in result.ranking
            if e.objective_score is None
        ]
        self.assertEqual(len(failed_ids), 4)

    def test_best_is_still_identified_after_partial_failure(self) -> None:
        adapter = FailingMockAdapter(fail_top_k=5)
        result = _make_engine(adapter=adapter).run()
        self.assertIsNotNone(result.best_configuration)
        self.assertNotEqual(result.best_configuration.parameters["top_k"], 5)

    def test_all_failures_raise_system_error(self) -> None:
        with self.assertRaises(SearchSystemError):
            _make_engine(adapter=AlwaysFailAdapter()).run()

    def test_failure_entries_have_error_status(self) -> None:
        adapter = FailingMockAdapter(fail_top_k=5)
        result = _make_engine(adapter=adapter).run()
        failed_entries = [
            e for e in result.ranking if e.objective_score is None
        ]
        self.assertTrue(
            all(e.status.startswith("CONFIG_ERROR") for e in failed_entries)
        )


class TestDeterministicBehavior(unittest.TestCase):
    def test_two_runs_produce_same_ranking(self) -> None:
        result_a = _make_engine().run()
        result_b = _make_engine().run()
        self.assertEqual(
            [e.to_dict() for e in result_a.ranking],
            [e.to_dict() for e in result_b.ranking],
        )

    def test_rank_entries_tie_break_is_deterministic(self) -> None:
        entries = [
            SearchRankingEntry("b", {"x": 2}, 0.9, 0.9, 0.9, 1.0, "SUCCESS"),
            SearchRankingEntry("a", {"x": 1}, 0.5, 0.5, 0.5, 2.0, "SUCCESS"),
            SearchRankingEntry("c", {"x": 3}, 0.5, 0.5, 0.5, 3.0, "SUCCESS"),
            SearchRankingEntry(
                "f", {"x": 4}, None, None, None, None, "CONFIG_ERROR: x"
            ),
        ]
        r1 = rank_entries(entries)
        r2 = rank_entries(entries)
        self.assertEqual(
            [e.experiment_id for e in r1],
            [e.experiment_id for e in r2],
        )
        self.assertEqual(r1[0].experiment_id, "b")
        self.assertEqual(r1[-1].experiment_id, "f")


class TestMockAdapterCompatibility(unittest.TestCase):
    def test_mock_adapter_satisfies_protocol(self) -> None:
        from src.adaptive_rag.system_adapter import SystemAdapter

        self.assertIsInstance(DeterministicMockAdapter(), SystemAdapter)


class TestLoadCases(unittest.TestCase):
    def test_enterprise_dataset_loads(self) -> None:
        cases = load_search_cases("enterprise", max_questions=2)
        self.assertEqual(len(cases), 2)
        self.assertIsInstance(cases[0], EvaluationCase)

    def test_default_dataset_loads(self) -> None:
        cases = load_search_cases("default", max_questions=3)
        self.assertEqual(len(cases), 3)

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_search_cases("does-not-exist")


if __name__ == "__main__":
    unittest.main()
