"""Tests for the deterministic retrieval-quality evaluator.

These tests are fully deterministic and require no API access.
"""

import unittest

from src.adaptive_rag.retrieval_evaluator import (
    RetrievalMetrics,
    calculate_context_relevance,
    calculate_retrieval_metrics,
    calculate_retrieval_metrics_at_k,
    normalize_doc_id,
)


class TestNormalizeDocId(unittest.TestCase):
    def test_normalize_strips_data_prefix(self) -> None:
        self.assertEqual(
            normalize_doc_id("data/supervised_learning.md"),
            "supervised_learning",
        )

    def test_normalize_strips_extension(self) -> None:
        self.assertEqual(
            normalize_doc_id("supervised_learning.md"),
            "supervised_learning",
        )

    def test_normalize_strips_docs_prefix(self) -> None:
        self.assertEqual(
            normalize_doc_id("docs/machine_learning.md"),
            "machine_learning",
        )

    def test_normalize_strips_dot_slash(self) -> None:
        self.assertEqual(
            normalize_doc_id("./machine_learning.md"),
            "machine_learning",
        )

    def test_normalize_lowercases(self) -> None:
        self.assertEqual(
            normalize_doc_id("Machine_Learning.md"),
            "machine_learning",
        )

    def test_normalize_empty_string(self) -> None:
        self.assertEqual(normalize_doc_id(""), "")

    def test_normalize_plain_name(self) -> None:
        self.assertEqual(normalize_doc_id("overfitting"), "overfitting")


class TestRetrievalMetrics(unittest.TestCase):
    def test_perfect_retrieval(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc2.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.retrieved_count, 2)
        self.assertEqual(metrics.relevant_count, 2)
        self.assertEqual(metrics.relevant_retrieved_count, 2)

    def test_zero_retrieval(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=[],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)
        self.assertEqual(metrics.retrieved_count, 0)
        self.assertEqual(metrics.relevant_count, 2)
        self.assertEqual(metrics.relevant_retrieved_count, 0)

    def test_partial_retrieval(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1, 0.5)
        self.assertEqual(metrics.retrieved_count, 2)
        self.assertEqual(metrics.relevant_count, 2)
        self.assertEqual(metrics.relevant_retrieved_count, 1)

    def test_irrelevant_retrieval(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc3.md", "doc4.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)
        self.assertEqual(metrics.relevant_retrieved_count, 0)

    def test_duplicate_retrieved_documents(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc1.md", "doc2.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.retrieved_count, 2)

    def test_no_relevant_documents(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md"],
            relevant_documents=[],
        )
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)
        self.assertEqual(metrics.relevant_count, 0)

    def test_normalization_of_document_paths(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["data/doc1.md"],
            relevant_documents=["doc1.md"],
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)

    def test_precision_calculation(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc2.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        # 2 relevant retrieved out of 3 retrieved
        self.assertAlmostEqual(metrics.precision, 2 / 3, places=5)

    def test_recall_calculation(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md", "doc2.md", "doc3.md"],
        )
        # 1 relevant retrieved out of 3 relevant
        self.assertAlmostEqual(metrics.recall, 1 / 3, places=5)

    def test_f1_calculation(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc2.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        # precision = 2/3, recall = 1.0
        # f1 = 2 * (2/3) * 1.0 / (2/3 + 1.0) = (4/3) / (5/3) = 4/5 = 0.8
        self.assertAlmostEqual(metrics.f1, 0.8, places=5)

    def test_scores_remain_between_0_and_1(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md", "doc2.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc4.md"],
        )
        for score in (metrics.precision, metrics.recall, metrics.f1):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestTopKEvaluation(unittest.TestCase):
    def test_top_1_evaluation(self) -> None:
        metrics = calculate_retrieval_metrics_at_k(
            retrieved_documents=["doc1.md", "doc2.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc2.md"],
            k=1,
        )
        # Only doc1 is in top-1, which is relevant
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.retrieved_count, 1)

    def test_top_3_evaluation(self) -> None:
        metrics = calculate_retrieval_metrics_at_k(
            retrieved_documents=["doc1.md", "doc2.md", "doc3.md"],
            relevant_documents=["doc1.md", "doc2.md"],
            k=3,
        )
        # doc1 and doc2 are relevant, doc3 is not
        self.assertEqual(metrics.precision, 2 / 3)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.retrieved_count, 3)

    def test_top_5_evaluation(self) -> None:
        metrics = calculate_retrieval_metrics_at_k(
            retrieved_documents=["doc1.md", "doc2.md"],
            relevant_documents=["doc1.md", "doc2.md"],
            k=5,
        )
        # k=5 but only 2 retrieved, so all are used
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.retrieved_count, 2)

    def test_top_k_exceeds_retrieved(self) -> None:
        metrics = calculate_retrieval_metrics_at_k(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md", "doc2.md"],
            k=10,
        )
        # k=10 but only 1 retrieved
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.5)


class TestContextRelevance(unittest.TestCase):
    def test_relevant_context_gets_high_score(self) -> None:
        score = calculate_context_relevance(
            question="What is machine learning?",
            retrieved_context="Machine learning is a field of study about algorithms.",
        )
        self.assertGreater(score, 0.5)

    def test_irrelevant_context_gets_low_score(self) -> None:
        score = calculate_context_relevance(
            question="What is machine learning?",
            retrieved_context="The weather is sunny and warm today.",
        )
        self.assertLess(score, 0.3)

    def test_no_context_returns_zero(self) -> None:
        score = calculate_context_relevance(
            question="What is machine learning?",
            retrieved_context=None,
        )
        self.assertEqual(score, 0.0)

    def test_empty_context_returns_zero(self) -> None:
        score = calculate_context_relevance(
            question="What is machine learning?",
            retrieved_context="",
        )
        self.assertEqual(score, 0.0)

    def test_context_relevance_stays_between_0_and_1(self) -> None:
        score = calculate_context_relevance(
            question="What is machine learning?",
            retrieved_context="Machine learning algorithms and models.",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestRetrievalMetricsSerialization(unittest.TestCase):
    def test_to_dict(self) -> None:
        metrics = calculate_retrieval_metrics(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md"],
        )
        d = metrics.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("precision", d)
        self.assertIn("recall", d)
        self.assertIn("f1", d)
        self.assertIn("retrieved_count", d)
        self.assertIn("relevant_count", d)
        self.assertIn("relevant_retrieved_count", d)


if __name__ == "__main__":
    unittest.main()
