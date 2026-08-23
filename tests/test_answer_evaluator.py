"""Tests for the baseline heuristic answer-quality evaluator.

These tests are fully deterministic and require no API access.
"""

import unittest

from src.adaptive_rag.answer_evaluator import (
    DEFAULT_ANSWER_WEIGHTS,
    AnswerScores,
    evaluate_answer,
    tokenize,
)
from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse


class TestTokenize(unittest.TestCase):
    def test_tokenize_lowercases_and_removes_stopwords(self) -> None:
        tokens = tokenize("The quick brown fox")
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)
        self.assertNotIn("the", tokens)

    def test_tokenize_handles_none(self) -> None:
        self.assertEqual(tokenize(None), [])

    def test_tokenize_handles_empty_string(self) -> None:
        self.assertEqual(tokenize(""), [])


class TestCorrectnessScore(unittest.TestCase):
    def test_exact_expected_answer_gets_high_correctness(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        self.assertGreater(scores.correctness_score, 0.9)

    def test_unrelated_answer_gets_low_correctness(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="The capital of France is Paris.")
        scores = evaluate_answer(case, response)
        self.assertLess(scores.correctness_score, 0.3)


class TestRelevanceScore(unittest.TestCase):
    def test_relevant_answer_gets_high_relevance(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        self.assertGreater(scores.relevance_score, 0.5)

    def test_unrelated_answer_gets_low_relevance(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="The weather is sunny today.")
        scores = evaluate_answer(case, response)
        self.assertLess(scores.relevance_score, 0.3)


class TestGroundingScore(unittest.TestCase):
    def test_grounded_answer_gets_high_grounding(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            retrieved_context="Machine learning is a field of study that involves algorithms.",
        )
        scores = evaluate_answer(case, response)
        self.assertGreater(scores.grounding_score, 0.7)

    def test_unsupported_answer_gets_low_grounding(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            retrieved_context="The weather is sunny and warm today.",
        )
        scores = evaluate_answer(case, response)
        self.assertLess(scores.grounding_score, 0.3)

    def test_no_context_gets_zero_grounding(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        self.assertEqual(scores.grounding_score, 0.0)


class TestCompletenessScore(unittest.TestCase):
    def test_complete_answer_gets_high_completeness(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        self.assertGreater(scores.completeness_score, 0.9)

    def test_incomplete_answer_gets_lower_completeness(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study about algorithms.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        self.assertLess(scores.completeness_score, 1.0)


class TestAnswerScoreRange(unittest.TestCase):
    def test_answer_score_stays_between_0_and_1(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            retrieved_context="Machine learning is a field of study.",
        )
        scores = evaluate_answer(case, response)
        self.assertGreaterEqual(scores.answer_score, 0.0)
        self.assertLessEqual(scores.answer_score, 1.0)

    def test_answer_score_with_unrelated_answer_stays_in_range(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(
            answer="The capital of France is Paris.",
            retrieved_context="The weather is sunny.",
        )
        scores = evaluate_answer(case, response)
        self.assertGreaterEqual(scores.answer_score, 0.0)
        self.assertLessEqual(scores.answer_score, 1.0)


class TestConfigurableWeights(unittest.TestCase):
    def test_custom_weights_are_used(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            retrieved_context="Machine learning is a field of study.",
        )

        default_scores = evaluate_answer(case, response)
        custom_weights = {
            "correctness": 0.10,
            "relevance": 0.30,
            "grounding": 0.30,
            "completeness": 0.30,
        }
        custom_scores = evaluate_answer(case, response, weights=custom_weights)

        self.assertEqual(custom_scores.weights, custom_weights)
        self.assertNotEqual(default_scores.weights, custom_weights)

    def test_default_weights_match_expected(self) -> None:
        self.assertEqual(DEFAULT_ANSWER_WEIGHTS["correctness"], 0.40)
        self.assertEqual(DEFAULT_ANSWER_WEIGHTS["relevance"], 0.25)
        self.assertEqual(DEFAULT_ANSWER_WEIGHTS["grounding"], 0.25)
        self.assertEqual(DEFAULT_ANSWER_WEIGHTS["completeness"], 0.10)


class TestAnswerScoresSerialization(unittest.TestCase):
    def test_answer_scores_to_dict(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
        )
        response = SystemResponse(answer="Machine learning is a field of study.")
        scores = evaluate_answer(case, response)
        d = scores.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("correctness_score", d)
        self.assertIn("relevance_score", d)
        self.assertIn("grounding_score", d)
        self.assertIn("completeness_score", d)
        self.assertIn("answer_score", d)
        self.assertIn("weights", d)


if __name__ == "__main__":
    unittest.main()
