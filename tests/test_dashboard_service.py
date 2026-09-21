"""Tests for the Phase 5 dashboard service (API-free, no Streamlit UI)."""

import unittest

from src.adaptive_rag.dashboard_service import (
    compare_search_methods,
    demo_questions,
    documents_by_category,
    get_recommended_configuration,
    list_configurations,
    load_benchmark_questions,
    load_enterprise_documents,
    load_rl_split,
    run_diagnosis,
    run_retrieval_evaluation,
    search_space_summary,
)


class TestDashboardLoading(unittest.TestCase):
    def test_documents(self):
        docs = load_enterprise_documents()
        self.assertEqual(len(docs), 25)

    def test_categories(self):
        cats = documents_by_category()
        self.assertEqual(set(cats), {"hr", "engineering", "finance",
                                     "product", "security"})
        self.assertEqual(sum(cats.values()), 25)

    def test_questions(self):
        questions = load_benchmark_questions()
        self.assertEqual(len(questions), 20)
        self.assertEqual(len(demo_questions()), 2)

    def test_split(self):
        split = load_rl_split()
        self.assertEqual(len(split["training_case_ids"]), 14)
        self.assertEqual(len(split["test_case_ids"]), 6)

    def test_search_space(self):
        space = search_space_summary()
        self.assertEqual(space["total_configurations"], 175)
        self.assertEqual(len(list_configurations()), 175)


class TestDashboardEvaluation(unittest.TestCase):
    def test_retrieval_evaluation_local(self):
        ev = run_retrieval_evaluation("ent-001", 300, 50, 3)
        self.assertIn("precision", ev)
        self.assertIn("retrieved_documents", ev)
        self.assertIsNotNone(ev["f1"])
        self.assertGreaterEqual(ev["f1"], 0.0)
        self.assertLessEqual(ev["f1"], 1.0)

    def test_diagnosis(self):
        ev = run_retrieval_evaluation("ent-001", 300, 50, 3)
        diag = run_diagnosis(ev)
        self.assertIn(diag["overall_status"], ("PASS", "ISSUES FOUND"))
        self.assertIsInstance(diag["findings"], list)

    def test_comparison_labelled(self):
        comp = compare_search_methods()
        self.assertIn("smoke-test", comp["label"])
        self.assertEqual(len(comp["rows"]), 4)
        self.assertIn(comp["best_method"], ("random", "adaptive"))

    def test_recommendation_measured(self):
        rec = get_recommended_configuration()
        self.assertEqual(rec["recommended"]["chunk_size"], 500)
        self.assertIn("smoke-test", rec["source"])


if __name__ == "__main__":
    unittest.main()
