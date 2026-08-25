"""Tests for the adaptive search result data model (Phase 3D)."""

import json
import unittest

from src.adaptive_rag.adaptive_search_result import (
    AdaptiveRankingEntry,
    AdaptiveSearchResult,
    EvaluationOrderEntry,
)


def _ranking_entry(eid="adaptive-001", score=0.75):
    return AdaptiveRankingEntry(
        experiment_id=eid,
        parameters={"chunk_size": 100, "chunk_overlap": 10, "top_k": 2},
        objective_score=score,
        retrieval_f1=0.9,
        context_relevance=0.8,
        latency_ms=42.5,
        status="SUCCESS",
    )


def _order_entry(eid="adaptive-001", reason="initial exploration"):
    return EvaluationOrderEntry(
        experiment_id=eid,
        parameters={"chunk_size": 100, "chunk_overlap": 10, "top_k": 2},
        objective_score=0.75,
        retrieval_f1=0.9,
        context_relevance=0.8,
        latency_ms=42.5,
        status="SUCCESS",
        selection_reason=reason,
    )


class TestCreation(unittest.TestCase):
    def test_default_creation(self) -> None:
        result = AdaptiveSearchResult()
        self.assertEqual(result.search_id, "")
        self.assertEqual(result.total_possible_configurations, 0)
        self.assertEqual(result.ranking, [])
        self.assertEqual(result.evaluation_order, [])
        self.assertIsNone(result.best_configuration)

    def test_full_creation(self) -> None:
        entry = _ranking_entry()
        result = AdaptiveSearchResult(
            search_id="s1",
            search_space={"name": "tiny"},
            total_possible_configurations=8,
            configurations_evaluated=1,
            configurations_remaining=7,
            best_configuration=entry,
            best_objective_score=entry.objective_score,
            ranking=[entry],
            evaluation_order=[_order_entry()],
            selection_reasons={"adaptive-001": "initial exploration"},
        )
        self.assertEqual(result.configurations_evaluated, 1)
        self.assertEqual(result.best_objective_score, 0.75)


class TestSerialization(unittest.TestCase):
    def test_ranking_entry_serialization(self) -> None:
        data = _ranking_entry().to_dict()
        self.assertEqual(data["experiment_id"], "adaptive-001")
        self.assertEqual(data["status"], "SUCCESS")
        json.dumps(data)  # must be JSON-serializable

    def test_order_entry_serialization(self) -> None:
        data = _order_entry().to_dict()
        self.assertIn("selection_reason", data)
        self.assertEqual(data["selection_reason"], "initial exploration")
        json.dumps(data)

    def test_to_json_is_valid_json(self) -> None:
        result = AdaptiveSearchResult(
            search_id="s1",
            total_possible_configurations=175,
            configurations_evaluated=2,
            configurations_remaining=173,
            ranking=[_ranking_entry()],
            evaluation_order=[_order_entry()],
        )
        parsed = json.loads(result.to_json())
        self.assertEqual(parsed["search_id"], "s1")
        self.assertEqual(parsed["total_possible_configurations"], 175)

    def test_round_trip_preserves_fields(self) -> None:
        result = AdaptiveSearchResult(
            search_id="rt",
            total_possible_configurations=8,
            configurations_evaluated=3,
            configurations_remaining=5,
            failed_configurations=1,
            best_configuration=_ranking_entry("adaptive-002", 0.81),
            best_objective_score=0.81,
            ranking=[_ranking_entry(), _ranking_entry("adaptive-002", 0.81)],
            evaluation_order=[
                _order_entry(),
                _order_entry("adaptive-002", "neighbour average"),
            ],
            selection_reasons={
                "adaptive-001": "initial exploration",
                "adaptive-002": "neighbour average",
            },
        )
        data = result.to_dict()
        from src.adaptive_rag.adaptive_search_result import (
            AdaptiveRankingEntry as ARE,
        )

        rebuilt = AdaptiveSearchResult(
            search_id=data["search_id"],
            search_space=data["search_space"],
            total_possible_configurations=data["total_possible_configurations"],
            configurations_evaluated=data["configurations_evaluated"],
            configurations_remaining=data["configurations_remaining"],
            failed_configurations=data["failed_configurations"],
            best_configuration=ARE(**data["best_configuration"]),
            best_objective_score=data["best_objective_score"],
            ranking=[ARE(**e) for e in data["ranking"]],
            evaluation_order=[EvaluationOrderEntry(**e) for e in data["evaluation_order"]],
            selection_reasons=dict(data["selection_reasons"]),
        )
        self.assertEqual(rebuilt.to_dict(), result.to_dict())


class TestRankingAndOrder(unittest.TestCase):
    def test_ranking_list_exposed(self) -> None:
        entries = [_ranking_entry("a"), _ranking_entry("b")]
        result = AdaptiveSearchResult(ranking=entries)
        self.assertEqual(len(result.ranking), 2)
        self.assertEqual([e.experiment_id for e in result.ranking], ["a", "b"])

    def test_evaluation_order_preserved(self) -> None:
        order = [
            _order_entry("adaptive-001", "initial exploration"),
            _order_entry("adaptive-002", "exploration bonus"),
            _order_entry("adaptive-003", "neighbour average objective 0.9000"),
        ]
        result = AdaptiveSearchResult(evaluation_order=order)
        reasons = [e.selection_reason for e in result.evaluation_order]
        self.assertEqual(reasons[0], "initial exploration")
        self.assertIn("neighbour average", reasons[2])


if __name__ == "__main__":
    unittest.main()