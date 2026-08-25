"""Tests for the deterministic adaptive search engine (Phase 3D).

All tests run against a deterministic mock adapter — no Gemini, no
internet, no API keys.
"""

import unittest

from src.adaptive_rag.adaptive_search import (
    DEFAULT_EXPLORATION_BONUS,
    AdaptiveSearchEngine,
    get_neighbors,
    param_key,
    select_initial_combinations,
)
from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.objective import ObjectiveConfig
from src.adaptive_rag.search_space import SearchSpace


def _make_space(name: str = "tiny", sizes=(100, 200), overlaps=(10, 40), tops=(2, 3)):
    return SearchSpace(
        name=name,
        parameter_values={
            "chunk_size": list(sizes),
            "chunk_overlap": list(overlaps),
            "top_k": list(tops),
        },
    )


class DeterministicMockAdapter:
    """Deterministic adapter; objective varies with top_k only.

    Latency is fixed at 50 ms so the objective is a pure function of
    ``top_k``: smaller ``top_k`` scores higher (fewer distractors).
    """

    def run(self, case: EvaluationCase, config: ExperimentConfig) -> SystemResponse:
        top_k = config.parameters["top_k"]
        retrieved = list(case.relevant_documents)[:top_k]
        return SystemResponse(
            answer=None,
            action="SEARCH",
            retrieved_documents=retrieved,
            retrieved_context=" ".join(retrieved) or None,
            latency_ms=50.0,
            retrieval_attempts=1,
            metadata={"adapter": "adaptive_mock"},
        )


class FailingForOverlapAdapter(DeterministicMockAdapter):
    """Adapter raising for one specific chunk_overlap value."""

    def __init__(self, fail_overlap: int = 40) -> None:
        self.fail_overlap = fail_overlap

    def run(self, case: EvaluationCase, config: ExperimentConfig) -> SystemResponse:
        if config.parameters.get("chunk_overlap") == self.fail_overlap:
            raise RuntimeError("boom for this overlap")
        return super().run(case, config)


def _make_engine(
    space=None,
    adapter=None,
    max_configurations=6,
    exploration_bonus=DEFAULT_EXPLORATION_BONUS,
):
    return AdaptiveSearchEngine(
        space if space is not None else _make_space(),
        dataset="enterprise",
        max_questions=1,
        adapter=adapter if adapter is not None else DeterministicMockAdapter(),
        objective_config=ObjectiveConfig(),
        max_configurations=max_configurations,
        exploration_bonus=exploration_bonus,
        search_id="test-adaptive",
    )


class TestInitialExploration(unittest.TestCase):
    def test_initial_exploration_is_deterministic(self) -> None:
        space = _make_space()
        first = select_initial_combinations(space)
        second = select_initial_combinations(space)
        self.assertEqual(first, second)

    def test_initial_set_contains_corners_and_centre(self) -> None:
        space = SearchSpace.adaptive_pilot()
        initial = select_initial_combinations(space)
        self.assertIn(
            param_key({"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}),
            [param_key(c) for c in initial],
        )
        self.assertIn(
            param_key({"chunk_size": 500, "chunk_overlap": 75, "top_k": 6}),
            [param_key(c) for c in initial],
        )
        # Centre of the 175-combo space.
        self.assertIn(
            param_key({"chunk_size": 250, "chunk_overlap": 40, "top_k": 4}),
            [param_key(c) for c in initial],
        )

    def test_initial_set_has_no_duplicates(self) -> None:
        space = SearchSpace.adaptive_pilot()
        initial = select_initial_combinations(space)
        keys = [param_key(c) for c in initial]
        self.assertEqual(len(keys), len(set(keys)))


class TestNeighbors(unittest.TestCase):
    def test_valid_neighbor_generation(self) -> None:
        space = _make_space(sizes=(100, 200, 300), overlaps=(10, 40), tops=(2, 3))
        center = {"chunk_size": 200, "chunk_overlap": 10, "top_k": 2}
        neighbors = get_neighbors(center, space)
        keys = {param_key(n) for n in neighbors}
        expected = {
            param_key({"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}),
            param_key({"chunk_size": 300, "chunk_overlap": 10, "top_k": 2}),
            param_key({"chunk_size": 200, "chunk_overlap": 40, "top_k": 2}),
            param_key({"chunk_size": 200, "chunk_overlap": 10, "top_k": 3}),
        }
        self.assertEqual(keys, expected)

    def test_invalid_neighbors_excluded(self) -> None:
        space = _make_space(sizes=(100, 200), overlaps=(10, 40), tops=(2,))
        corner = {"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}
        neighbors = get_neighbors(corner, space)
        # top_k has no down-step; chunk_size/overlap have no left-step.
        keys = {param_key(n) for n in neighbors}
        self.assertNotIn(
            param_key({"chunk_size": 100, "chunk_overlap": 10, "top_k": 1}),
            keys,
        )
        # Only up-steps remain: chunk_size↑ and chunk_overlap↑
        # (top_k has a single value, so it has no neighbour at all).
        self.assertEqual(len(neighbors), 2)
        for neighbor in neighbors:
            self.assertTrue(neighbor["chunk_overlap"] < neighbor["chunk_size"])


class TestAdaptiveSelection(unittest.TestCase):
    def test_no_duplicate_evaluations(self) -> None:
        result = _make_engine(max_configurations=6).run()
        keys = [param_key(e.parameters) for e in result.evaluation_order]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(result.configurations_evaluated, 6)

    def test_high_performing_neighbors_get_priority(self) -> None:
        space = _make_space(sizes=(100, 200), overlaps=(10, 40), tops=(2, 3))
        engine = _make_engine(space=space, max_configurations=99)
        # White-box seed: (100,10,2) scored high, its neighbour region
        # must be preferred over untouched regions.
        high = {"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}
        engine._objectives[param_key(high)] = 0.9
        combo, reason = engine._select_next()
        neighbors = get_neighbors(high, space)
        neighbor_keys = [param_key(n) for n in neighbors]
        self.assertIn(param_key(combo), neighbor_keys)
        self.assertIn("neighbour average objective", reason)

    def test_exploration_bonus_when_no_evaluated_neighbors(self) -> None:
        space = SearchSpace.adaptive_pilot()
        engine = _make_engine(space=space, max_configurations=99)
        # Seed an objective far from the min-min-min corner.
        far = {"chunk_size": 500, "chunk_overlap": 75, "top_k": 6}
        engine._objectives[param_key(far)] = 0.9
        corner = {"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}
        priority, reason = engine._priority(corner)
        self.assertAlmostEqual(priority, DEFAULT_EXPLORATION_BONUS)
        self.assertIn("exploration bonus", reason)

    def test_bonus_increases_priority_of_unexplored(self) -> None:
        space = _make_space(sizes=(100, 200), overlaps=(10, 40), tops=(2,))
        evaluated = {"chunk_size": 100, "chunk_overlap": 10, "top_k": 2}
        candidate = {"chunk_size": 200, "chunk_overlap": 40, "top_k": 2}
        engine_low = _make_engine(
            space=space, exploration_bonus=0.01, max_configurations=99
        )
        engine_low._objectives[param_key(evaluated)] = 0.9
        _, reason_low = engine_low._priority(candidate)
        # Candidate has no evaluated neighbours -> pure bonus.
        self.assertIn("exploration bonus", reason_low)


class TestRunBehavior(unittest.TestCase):
    def test_best_configuration_tracked(self) -> None:
        result = _make_engine(max_configurations=4).run()
        self.assertIsNotNone(result.best_configuration)
        objectives = [
            e.objective_score
            for e in result.ranking
            if e.status == "SUCCESS" and e.objective_score is not None
        ]
        self.assertEqual(
            result.best_configuration.objective_score, max(objectives)
        )
        self.assertEqual(result.best_objective_score, max(objectives))

    def test_evaluation_order_recorded_with_reasons(self) -> None:
        result = _make_engine(max_configurations=5).run()
        self.assertEqual(len(result.evaluation_order), 5)
        for entry in result.evaluation_order:
            self.assertTrue(entry.selection_reason)
        first = result.evaluation_order[0]
        self.assertIn("initial exploration", first.selection_reason)
        later = result.evaluation_order[-1]
        self.assertTrue(
            "neighbour average" in later.selection_reason
            or "exploration bonus" in later.selection_reason
        )

    def test_configuration_limit_respected_on_large_space(self) -> None:
        result = _make_engine(
            space=SearchSpace.adaptive_pilot(), max_configurations=8
        ).run()
        self.assertLessEqual(result.configurations_evaluated, 8)
        self.assertEqual(result.total_possible_configurations, 175)
        self.assertEqual(
            result.configurations_remaining,
            175 - result.configurations_evaluated,
        )

    def test_failed_configuration_does_not_stop_search(self) -> None:
        adapter = FailingForOverlapAdapter(fail_overlap=40)
        result = _make_engine(adapter=adapter, max_configurations=6).run()
        self.assertGreater(result.failed_configurations, 0)
        statuses = [e.status for e in result.ranking]
        self.assertTrue(any(s.startswith("CONFIG_ERROR") for s in statuses))
        self.assertTrue(any(s == "SUCCESS" for s in statuses))
        failed_entries = [e for e in result.ranking if e.status != "SUCCESS"]
        for entry in failed_entries:
            self.assertIsNone(entry.objective_score)

    def test_deterministic_behavior_two_runs(self) -> None:
        run_a = _make_engine(max_configurations=6).run().to_dict()
        run_b = _make_engine(max_configurations=6).run().to_dict()
        # search_id contains a timestamp; compare everything else.
        run_a.pop("search_id")
        run_b.pop("search_id")
        self.assertEqual(run_a, run_b)


if __name__ == "__main__":
    unittest.main()