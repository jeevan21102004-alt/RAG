"""Tests for grid-vs-adaptive search comparison (Phase 3D)."""

import unittest

from src.adaptive_rag.adaptive_search_result import (
    AdaptiveRankingEntry,
    AdaptiveSearchResult,
)
from src.adaptive_rag.search_comparison import (
    SearchComparison,
    compare_grid_vs_adaptive,
    summarize_search,
)
from src.adaptive_rag.search_result import SearchRankingEntry, SearchResult


def _grid_entry(eid, score, f1, ctx, latency, status="SUCCESS"):
    return SearchRankingEntry(
        experiment_id=eid,
        parameters={"chunk_size": 100},
        objective_score=score,
        retrieval_f1=f1,
        context_relevance=ctx,
        latency_ms=latency,
        status=status,
    )


def _adaptive_entry(eid, score, f1, ctx, latency, status="SUCCESS"):
    return AdaptiveRankingEntry(
        experiment_id=eid,
        parameters={"chunk_size": 100},
        objective_score=score,
        retrieval_f1=f1,
        context_relevance=ctx,
        latency_ms=latency,
        status=status,
    )


class TestSummarizeSearch(unittest.TestCase):
    def test_efficiency_and_coverage(self) -> None:
        ranking = [
            _adaptive_entry("a", 0.6, 0.7, 0.8, 50.0),
            _adaptive_entry("b", 0.8, 0.9, 0.85, 40.0),
        ]
        summary = summarize_search("adaptive", 4, ranking, 175)
        self.assertEqual(summary.configurations_evaluated, 4)
        self.assertAlmostEqual(summary.best_objective, 0.8)
        self.assertAlmostEqual(summary.search_efficiency, 0.2)  # 0.8 / 4
        self.assertAlmostEqual(summary.space_coverage, 4 / 175)
        self.assertAlmostEqual(summary.best_f1, 0.9)
        self.assertAlmostEqual(summary.fastest_latency_ms, 40.0)

    def test_missing_metrics_reported_as_none(self) -> None:
        summary = summarize_search("grid", 0, [], 175)
        self.assertIsNone(summary.best_objective)
        self.assertIsNone(summary.search_efficiency)
        # Zero evaluations over a non-empty space: coverage is defined
        # and equals 0.0.
        self.assertAlmostEqual(summary.space_coverage, 0.0)
        self.assertEqual(summary.configurations_evaluated, 0)

    def test_coverage_none_when_space_empty(self) -> None:
        summary = summarize_search("grid", 0, [], 0)
        self.assertIsNone(summary.space_coverage)

    def test_failures_excluded_from_best(self) -> None:
        ranking = [
            _adaptive_entry("ok", 0.5, 0.5, 0.5, 60.0),
            _adaptive_entry("bad", None, None, None, None, "CONFIG_ERROR: x"),
        ]
        summary = summarize_search("adaptive", 2, ranking, 10)
        self.assertAlmostEqual(summary.best_objective, 0.5)
        self.assertEqual(summary.failed_configurations, 1)


class TestGridVsAdaptive(unittest.TestCase):
    def _grid_result(self) -> SearchResult:
        return SearchResult(
            search_id="g1",
            total_configurations=175,
            completed_configurations=8,
            failed_configurations=0,
            ranking=[
                _grid_entry(f"grid-{i}", 0.5 + i * 0.01, 0.7, 0.8, 80.0 + i)
                for i in range(8)
            ],
            best_configuration=_grid_entry("grid-7", 0.57, 0.7, 0.8, 87.0),
            best_objective_score=0.57,
        )

    def _adaptive_result(self) -> AdaptiveSearchResult:
        entries = [
            _adaptive_entry(f"adaptive-{i:03d}", 0.55 + i * 0.02, 0.75, 0.82, 45.0 - i)
            for i in range(8)
        ]
        best = entries[-1]
        return AdaptiveSearchResult(
            search_id="a1",
            total_possible_configurations=175,
            configurations_evaluated=8,
            configurations_remaining=167,
            best_configuration=best,
            best_objective_score=best.objective_score,
            ranking=entries,
        )

    def test_comparison_structure(self) -> None:
        comparison = compare_grid_vs_adaptive(self._grid_result(), self._adaptive_result())
        self.assertIsInstance(comparison, SearchComparison)
        self.assertEqual(comparison.total_possible_configurations, 175)
        self.assertIsNotNone(comparison.grid)
        self.assertIsNotNone(comparison.adaptive)

    def test_efficiency_calculation(self) -> None:
        comparison = compare_grid_vs_adaptive(self._grid_result(), self._adaptive_result())
        # Grid best objective = 0.5 + 7*0.01 = 0.57 over 8 evaluated.
        self.assertAlmostEqual(comparison.grid.search_efficiency, 0.57 / 8)
        adaptive_best = 0.55 + 7 * 0.02
        self.assertAlmostEqual(
            comparison.adaptive.search_efficiency, adaptive_best / 8
        )

    def test_space_coverage_fraction(self) -> None:
        comparison = compare_grid_vs_adaptive(self._grid_result(), self._adaptive_result())
        expected = 8 / 175
        self.assertAlmostEqual(comparison.grid.space_coverage, expected)
        self.assertAlmostEqual(comparison.adaptive.space_coverage, expected)

    def test_deterministic_comparison(self) -> None:
        first = compare_grid_vs_adaptive(
            self._grid_result(), self._adaptive_result()
        ).to_dict()
        second = compare_grid_vs_adaptive(
            self._grid_result(), self._adaptive_result()
        ).to_dict()
        first.pop("metadata")
        second.pop("metadata")
        self.assertEqual(first, second)

    def test_missing_metrics_handled(self) -> None:
        empty_grid = SearchResult(search_id="empty")
        empty_adaptive = AdaptiveSearchResult(search_id="empty-a")
        comparison = compare_grid_vs_adaptive(empty_grid, empty_adaptive)
        self.assertIsNone(comparison.grid.best_objective)
        self.assertIsNone(comparison.adaptive.search_efficiency)


if __name__ == "__main__":
    unittest.main()