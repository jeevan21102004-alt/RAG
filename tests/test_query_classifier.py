"""Tests for the deterministic Phase 6A query classifier.

Zero API calls, zero network, zero RL training. Pure stdlib + unittest.
"""
import json
import unittest
from pathlib import Path

from src.adaptive_rag.query_classifier import (
    QueryFeatures,
    QueryType,
    classify_all,
    classify_benchmark_question,
    classify_features,
    classify_query,
    extract_query_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"


class TestFeatureExtraction(unittest.TestCase):
    def test_basic_counts(self):
        feats = extract_query_features(
            "What is the password expiration period?",
            category="enterprise_security",
            relevant_documents=["password_policy.md"],
            case_id="ent-017",
        )
        self.assertEqual(feats.case_id, "ent-017")
        self.assertEqual(feats.num_relevant_documents, 1)
        self.assertFalse(feats.has_multi_suffix)
        self.assertGreater(feats.word_count, 0)
        self.assertEqual(feats.char_count, len("What is the password expiration period?"))
        self.assertIn("security", feats.domains_mentioned)

    def test_multi_suffix_and_conjunction(self):
        feats = extract_query_features(
            "What is the hotel cap and what expense amount requires approval?",
            category="enterprise_finance_multi",
            relevant_documents=["travel_policy.md", "expense_policy.md"],
        )
        self.assertTrue(feats.has_multi_suffix)
        self.assertEqual(feats.num_relevant_documents, 2)
        self.assertGreaterEqual(feats.conjunction_count, 1)

    def test_edge_cases_coerced(self):
        feats = extract_query_features(None, category=None,
                                       relevant_documents=None, case_id=None)
        self.assertEqual(feats.question, "")
        self.assertEqual(feats.word_count, 0)
        self.assertEqual(feats.num_relevant_documents, 0)
        self.assertIsInstance(feats, QueryFeatures)

    def test_deterministic_extraction(self):
        kwargs = dict(question="Who must approve every production deployment?",
                      category="enterprise_engineering",
                      relevant_documents=["deployment_guidelines.md"])
        self.assertEqual(extract_query_features(**kwargs),
                         extract_query_features(**kwargs))


class TestQueryTypes(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(
            classify_query("How many days of paid annual leave per year?",
                           "enterprise_hr", ["leave_policy.md"]),
            QueryType.SIMPLE)

    def test_multi_document_by_doc_count(self):
        self.assertEqual(
            classify_query("Hotel cap and expense approval?",
                           "enterprise_finance",
                           ["travel_policy.md", "expense_policy.md"]),
            QueryType.MULTI_DOCUMENT)

    def test_multi_document_by_suffix(self):
        self.assertEqual(
            classify_query("Something generic?",
                           "enterprise_product_multi", ["only_one.md"]),
            QueryType.MULTI_DOCUMENT)

    def test_cross_domain(self):
        # 'leave' (hr) + 'password' (security) => two domains, single doc.
        self.assertEqual(
            classify_query(
                "What is the leave policy and password expiration period?",
                "enterprise_hr", ["leave_policy.md"]),
            QueryType.CROSS_DOMAIN)

    def test_technical(self):
        self.assertEqual(
            classify_query("What is the p95 latency target for API endpoints?",
                           "enterprise_engineering", ["api_standards.md"]),
            QueryType.TECHNICAL)

    def test_priority_multi_over_technical(self):
        feats = extract_query_features(
            "What is the password expiration and MDM enrollment for devices?",
            "enterprise_security_multi",
            ["password_policy.md", "device_security.md"])
        self.assertEqual(classify_features(feats), QueryType.MULTI_DOCUMENT)

    def test_repeated_classification_deterministic(self):
        for _ in range(5):
            self.assertEqual(
                classify_query("What is the price of the professional plan?",
                               "enterprise_product", ["pricing_policy.md"]),
                QueryType.SIMPLE)


class TestBenchmarkCoverage(unittest.TestCase):
    def test_all_benchmark_questions_classifiable(self):
        items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(items), 20)
        results = classify_all(items)
        self.assertEqual(len(results), 20)
        for item in items:
            self.assertIn(item["case_id"], results)
            self.assertIsInstance(results[item["case_id"]], QueryType)
            # single-row helper agrees with batch helper
            self.assertEqual(results[item["case_id"]],
                             classify_benchmark_question(item))

    def test_known_multi_document_ids(self):
        items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        results = classify_all(items)
        for cid in ("ent-018", "ent-019", "ent-020"):
            self.assertEqual(results[cid], QueryType.MULTI_DOCUMENT)

    def test_invalid_row_raises(self):
        with self.assertRaises(TypeError):
            classify_benchmark_question(["not", "a", "mapping"])


if __name__ == "__main__":
    unittest.main()
