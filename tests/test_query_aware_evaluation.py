"""Tests for the Phase 6B query-aware evaluation pilot (retrieval-only).

Local execution only: no Gemini, no network, no RL training.
The full pilot is 4 configurations x 20 questions; live evaluations are
limited to tiny overrides so this suite stays fast.
"""

from __future__ import annotations

import unittest

from src.adaptive_rag.query_aware_evaluation import (
    PILOT_CONFIGURATIONS,
    VALID_QUERY_TYPES,
    aggregate_metrics,
    classify_questions,
    compare_global_vs_query_specific,
    evaluate_configuration,
    group_by_query_type,
    load_questions,
    pilot_configurations,
    run_pilot,
    select_best,
)


class TestQuestionGrouping(unittest.TestCase):
    def test_every_question_gets_valid_type(self):
        labels = classify_questions()
        self.assertEqual(len(labels), 20)
        for label in labels.values():
            self.assertIn(label, VALID_QUERY_TYPES)

    def test_grouping_covers_all_questions(self):
        groups = group_by_query_type()
        total = sum(len(qs) for qs in groups.values())
        self.assertEqual(total, 20)
        for qtype in groups:
            self.assertIn(qtype, VALID_QUERY_TYPES)


class TestPilotEvaluation(unittest.TestCase):
    def test_small_live_evaluation_uses_real_metrics(self):
        result = evaluate_configuration(
            {"chunk_size": 300, "chunk_overlap": 50, "top_k": 3},
            load_questions()[:2],
        )
        self.assertEqual(len(result["evaluations"]), 2)
        for ev in result["evaluations"]:
            self.assertIn("f1", ev)
            self.assertIn("latency_ms", ev)
        self.assertIsNotNone(result["aggregated"]["objective_score"])

    def test_best_selection_is_deterministic(self):
        first = evaluate_configuration(dict(PILOT_CONFIGURATIONS[0]),
                                       load_questions()[:2])
        second = evaluate_configuration(dict(PILOT_CONFIGURATIONS[1]),
                                        load_questions()[:2])
        best_a = select_best([first, second])
        best_b = select_best([second, first])
        self.assertEqual(best_a["configuration"], best_b["configuration"])

    def test_tie_breaks_to_smallest_parameters(self):
        made = lambda cfg, obj: {"configuration": dict(cfg),
                                 "aggregated": {"objective_score": obj}}
        best = select_best([made({"chunk_size": 500, "chunk_overlap": 10,
                                  "top_k": 2}, 0.7),
                            made({"chunk_size": 100, "chunk_overlap": 20,
                                  "top_k": 3}, 0.7)])
        self.assertEqual(best["configuration"]["chunk_size"], 100)

    def test_pilot_comparison_is_deterministic(self):
        kwargs = {"configurations": [dict(PILOT_CONFIGURATIONS[0]),
                                     dict(PILOT_CONFIGURATIONS[1])],
                  "questions": load_questions()[:3]}
        run_a = run_pilot(**kwargs)
        run_b = run_pilot(**kwargs)
        self.assertEqual(run_a["budget"]["retrieval_evaluations"], 6)
        self.assertEqual(
            run_a["global_best"]["configuration"],
            run_b["global_best"]["configuration"])
        # Objective embeds measured retrieval latency, which naturally
        # varies a few ms run to run; the *selection* is deterministic,
        # the raw latency number is not. Compare with tolerance.
        self.assertAlmostEqual(
            run_a["global_best"]["aggregated"]["objective_score"],
            run_b["global_best"]["aggregated"]["objective_score"],
            delta=0.05)
        comparison = compare_global_vs_query_specific(
            run_a["per_type_best"], run_a["global_best"])
        self.assertIn("global", comparison)
        self.assertIn("difference", comparison)

    def test_aggregate_handles_empty_input_without_fabrication(self):
        agg = aggregate_metrics([])
        self.assertEqual(agg["n"], 0)
        self.assertIsNone(agg["objective_score"])


if __name__ == "__main__":
    unittest.main()
