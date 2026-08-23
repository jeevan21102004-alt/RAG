"""Tests for the evaluation runner.

These tests are fully deterministic and require no API access.
"""

import unittest

from src.adaptive_rag.evaluation_runner import (
    DEFAULT_OVERALL_WEIGHTS,
    calculate_decision_score,
    calculate_overall_score,
    calculate_retrieval_score,
    run_evaluation,
)
from src.adaptive_rag.evaluation_schema import EvaluationCase, EvaluationResult, SystemResponse


class TestRetrievalScore(unittest.TestCase):
    def test_perfect_retrieval(self) -> None:
        score = calculate_retrieval_score(
            retrieved_documents=["doc1.md", "doc2.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(score, 1.0)

    def test_partial_retrieval(self) -> None:
        score = calculate_retrieval_score(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(score, 0.5)

    def test_no_relevant_documents_returns_none(self) -> None:
        score = calculate_retrieval_score(
            retrieved_documents=["doc1.md"],
            relevant_documents=[],
        )
        self.assertIsNone(score)

    def test_no_retrieved_documents_returns_zero(self) -> None:
        score = calculate_retrieval_score(
            retrieved_documents=[],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        self.assertEqual(score, 0.0)

    def test_missing_retrieval_information(self) -> None:
        score = calculate_retrieval_score(
            retrieved_documents=[],
            relevant_documents=[],
        )
        self.assertIsNone(score)


class TestDecisionScore(unittest.TestCase):
    def test_decision_match(self) -> None:
        score = calculate_decision_score("SEARCH", "SEARCH")
        self.assertEqual(score, 1.0)

    def test_decision_mismatch(self) -> None:
        score = calculate_decision_score("SEARCH", "ANSWER")
        self.assertEqual(score, 0.0)

    def test_missing_expected_action_returns_none(self) -> None:
        score = calculate_decision_score(None, "SEARCH")
        self.assertIsNone(score)

    def test_missing_actual_action_returns_none(self) -> None:
        score = calculate_decision_score("SEARCH", None)
        self.assertIsNone(score)


class TestOverallScore(unittest.TestCase):
    def test_overall_score_remains_between_0_and_1(self) -> None:
        score = calculate_overall_score(0.8, 0.5, 1.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_overall_score_with_all_components(self) -> None:
        score = calculate_overall_score(0.8, 0.5, 1.0)
        expected = 0.8 * 0.60 + 0.5 * 0.25 + 1.0 * 0.15
        self.assertAlmostEqual(score, expected, places=5)

    def test_overall_score_redistributes_when_retrieval_missing(self) -> None:
        score = calculate_overall_score(0.8, None, 1.0)
        # answer weight 0.60, decision weight 0.15 → total 0.75
        expected = (0.8 * 0.60 + 1.0 * 0.15) / 0.75
        self.assertAlmostEqual(score, expected, places=5)

    def test_overall_score_all_none_returns_none(self) -> None:
        score = calculate_overall_score(None, None, None)
        self.assertIsNone(score)

    def test_overall_score_custom_weights(self) -> None:
        custom = {"answer": 0.5, "retrieval": 0.3, "decision": 0.2}
        score = calculate_overall_score(0.8, 0.5, 1.0, weights=custom)
        expected = 0.8 * 0.5 + 0.5 * 0.3 + 1.0 * 0.2
        self.assertAlmostEqual(score, expected, places=5)


class TestRunEvaluation(unittest.TestCase):
    def _make_case(self) -> EvaluationCase:
        return EvaluationCase(
            case_id="case-001",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )

    def test_complete_successful_evaluation(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
            latency_ms=100.0,
            retrieval_attempts=1,
        )
        result = run_evaluation(case, response)

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.case_id, "case-001")
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNotNone(result.answer_score)
        self.assertIsNotNone(result.retrieval_score)
        self.assertIsNotNone(result.decision_score)
        self.assertIsNotNone(result.overall_score)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_decision_match(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertEqual(result.decision_score, 1.0)

    def test_decision_mismatch(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="ANSWER",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertEqual(result.decision_score, 0.0)

    def test_perfect_retrieval(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertEqual(result.retrieval_score, 1.0)

    def test_partial_retrieval(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["other_doc.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertEqual(result.retrieval_score, 0.0)

    def test_no_relevant_documents(self) -> None:
        case = EvaluationCase(
            case_id="case-002",
            question="What is 2 + 2?",
            category="retrieval_not_required",
            expected_answer="Four.",
            expected_action="ANSWER",
            relevant_documents=[],
        )
        response = SystemResponse(
            answer="Four.",
            action="ANSWER",
            retrieved_documents=[],
        )
        result = run_evaluation(case, response)
        self.assertIsNone(result.retrieval_score)
        self.assertEqual(result.decision_score, 1.0)
        self.assertIsNotNone(result.overall_score)

    def test_missing_retrieval_information(self) -> None:
        case = EvaluationCase(
            case_id="case-003",
            question="What is 2 + 2?",
            category="retrieval_not_required",
            expected_answer="Four.",
            expected_action="ANSWER",
            relevant_documents=[],
        )
        response = SystemResponse(
            answer="Four.",
            action="ANSWER",
        )
        result = run_evaluation(case, response)
        self.assertIsNone(result.retrieval_score)
        self.assertEqual(result.decision_score, 1.0)
        self.assertIsNotNone(result.overall_score)

    def test_overall_score_remains_between_0_and_1(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_metadata_contains_component_scores(self) -> None:
        case = self._make_case()
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
        )
        result = run_evaluation(case, response)
        self.assertIn("answer_scores", result.metadata)
        self.assertIn("retrieval_score", result.metadata)
        self.assertIn("decision_score", result.metadata)
        self.assertIn("overall_weights", result.metadata)

    def test_default_overall_weights(self) -> None:
        self.assertEqual(DEFAULT_OVERALL_WEIGHTS["answer"], 0.60)
        self.assertEqual(DEFAULT_OVERALL_WEIGHTS["retrieval"], 0.25)
        self.assertEqual(DEFAULT_OVERALL_WEIGHTS["decision"], 0.15)


if __name__ == "__main__":
    unittest.main()
