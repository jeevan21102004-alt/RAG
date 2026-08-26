"""Tests for the optimization result data model (Phase 4)."""

import json
import unittest

from src.adaptive_rag.optimization_result import (
    OptimizationEntry,
    OptimizationResult,
)


class TestOptimizationResult(unittest.TestCase):
    def test_entry_serialization(self) -> None:
        entry = OptimizationEntry(
            method="rl",
            step=1,
            action_id=7,
            parameters={"chunk_size": 100, "chunk_overlap": 10, "top_k": 2},
            objective_score=0.8,
            retrieval_f1=0.9,
            context_relevance=0.85,
            latency_ms=42.0,
            reward=0.9,
        )
        data = entry.to_dict()
        self.assertEqual(data["method"], "rl")
        self.assertEqual(data["action_id"], 7)
        json.dumps(data)

    def test_result_round_trip(self) -> None:
        entry = OptimizationEntry(
            method="grid",
            step=1,
            action_id=None,
            parameters={"chunk_size": 100, "chunk_overlap": 10, "top_k": 2},
            objective_score=0.7,
            retrieval_f1=0.8,
            context_relevance=0.75,
            latency_ms=50.0,
        )
        result = OptimizationResult(
            method="grid",
            seed=42,
            budget=4,
            search_space={"name": "adaptive-pilot"},
            configurations_evaluated=1,
            evaluation_order=[entry],
            best_configuration=entry,
            best_objective=0.7,
            best_f1=0.8,
            best_context_relevance=0.75,
            best_latency_ms=50.0,
        )
        data = json.loads(result.to_json())
        self.assertEqual(data["method"], "grid")
        self.assertEqual(data["budget"], 4)
        self.assertEqual(len(data["evaluation_order"]), 1)

    def test_comparison_metrics(self) -> None:
        result = OptimizationResult(
            method="random",
            budget=4,
            best_objective=0.6,
            best_f1=0.7,
            configurations_evaluated=4,
        )
        m = result.comparison_metrics()
        self.assertEqual(m["method"], "random")
        self.assertEqual(m["best_objective"], 0.6)
        self.assertEqual(m["configurations_evaluated"], 4)

    def test_defaults(self) -> None:
        result = OptimizationResult()
        self.assertEqual(result.method, "")
        self.assertIsNone(result.best_objective)
        self.assertEqual(result.evaluation_order, [])


if __name__ == "__main__":
    unittest.main()