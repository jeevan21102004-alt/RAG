"""Tests for the deterministic AI failure diagnosis engine.

These tests are fully deterministic and require no API access.
"""

import unittest

from src.adaptive_rag.diagnostics import (
    DEFAULT_THRESHOLDS,
    DiagnosticFinding,
    FailureType,
    Severity,
    SystemDiagnosis,
    diagnose_result,
    diagnose_system,
)
from src.adaptive_rag.evaluation_schema import EvaluationCase, EvaluationResult, SystemResponse
from src.adaptive_rag.evaluation_runner import run_evaluation


def _make_result(
    *,
    case_id: str = "case-001",
    question: str = "What is machine learning?",
    category: str = "retrieval_required",
    expected_answer: str | None = "Machine learning is a field of study.",
    expected_action: str | None = "SEARCH",
    relevant_documents: list[str] | None = None,
    answer: str | None = "Machine learning is a field of study.",
    action: str | None = "SEARCH",
    retrieved_documents: list[str] | None = None,
    retrieved_context: str | None = "Machine learning is a field of study.",
    latency_ms: float | None = 100.0,
    retrieval_attempts: int = 1,
) -> EvaluationResult:
    """Helper to build an EvaluationResult via the evaluation runner."""
    case = EvaluationCase(
        case_id=case_id,
        question=question,
        category=category,
        expected_answer=expected_answer,
        expected_action=expected_action,
        relevant_documents=relevant_documents or ["machine_learning_intro.md"],
    )
    response = SystemResponse(
        answer=answer,
        action=action,
        retrieved_documents=retrieved_documents or ["machine_learning_intro.md"],
        retrieved_context=retrieved_context,
        latency_ms=latency_ms,
        retrieval_attempts=retrieval_attempts,
    )
    return run_evaluation(case, response)


class TestLowRecall(unittest.TestCase):
    def test_low_recall_produces_retrieval_failure(self) -> None:
        result = _make_result(
            retrieved_documents=["other_doc.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        findings = diagnose_result(result)
        recall_findings = [
            f for f in findings if f.failure_type == FailureType.RETRIEVAL_LOW_RECALL
        ]
        self.assertEqual(len(recall_findings), 1)
        self.assertLess(recall_findings[0].score, 0.70)


class TestLowPrecision(unittest.TestCase):
    def test_low_precision_produces_precision_failure(self) -> None:
        result = _make_result(
            retrieved_documents=["doc1.md", "other1.md", "other2.md"],
            relevant_documents=["doc1.md"],
        )
        findings = diagnose_result(result)
        precision_findings = [
            f for f in findings if f.failure_type == FailureType.RETRIEVAL_LOW_PRECISION
        ]
        self.assertEqual(len(precision_findings), 1)
        self.assertLess(precision_findings[0].score, 0.70)


class TestLowContextRelevance(unittest.TestCase):
    def test_low_context_relevance_produces_context_failure(self) -> None:
        result = _make_result(
            retrieved_context="The weather is sunny and warm today.",
        )
        findings = diagnose_result(result)
        context_findings = [
            f for f in findings if f.failure_type == FailureType.CONTEXT_LOW_RELEVANCE
        ]
        self.assertEqual(len(context_findings), 1)
        self.assertLess(context_findings[0].score, 0.60)


class TestLowCorrectness(unittest.TestCase):
    def test_low_correctness_produces_answer_failure(self) -> None:
        result = _make_result(
            expected_answer="Machine learning is a field of study.",
            answer="The capital of France is Paris.",
        )
        findings = diagnose_result(result)
        correctness_findings = [
            f for f in findings if f.failure_type == FailureType.ANSWER_LOW_CORRECTNESS
        ]
        self.assertEqual(len(correctness_findings), 1)
        self.assertLess(correctness_findings[0].score, 0.70)


class TestLowGrounding(unittest.TestCase):
    def test_low_grounding_produces_grounding_failure(self) -> None:
        result = _make_result(
            answer="Machine learning is a field of study.",
            retrieved_context="The weather is sunny and warm today.",
        )
        findings = diagnose_result(result)
        grounding_findings = [
            f for f in findings if f.failure_type == FailureType.ANSWER_LOW_GROUNDING
        ]
        self.assertEqual(len(grounding_findings), 1)
        self.assertLess(grounding_findings[0].score, 0.70)


class TestLowCompleteness(unittest.TestCase):
    def test_low_completeness_produces_completeness_failure(self) -> None:
        result = _make_result(
            expected_answer="Machine learning is a field of study about algorithms and models.",
            answer="Machine learning is a field of study.",
        )
        findings = diagnose_result(result)
        completeness_findings = [
            f for f in findings if f.failure_type == FailureType.ANSWER_LOW_COMPLETENESS
        ]
        self.assertEqual(len(completeness_findings), 1)
        self.assertLess(completeness_findings[0].score, 0.70)


class TestDecisionError(unittest.TestCase):
    def test_decision_mismatch_produces_decision_failure(self) -> None:
        result = _make_result(
            expected_action="SEARCH",
            action="ANSWER",
        )
        findings = diagnose_result(result)
        decision_findings = [
            f for f in findings if f.failure_type == FailureType.DECISION_ERROR
        ]
        self.assertEqual(len(decision_findings), 1)
        self.assertEqual(decision_findings[0].score, 0.0)


class TestHighLatency(unittest.TestCase):
    def test_high_latency_produces_latency_failure(self) -> None:
        result = _make_result(latency_ms=3000.0)
        findings = diagnose_result(result)
        latency_findings = [
            f for f in findings if f.failure_type == FailureType.HIGH_LATENCY
        ]
        self.assertEqual(len(latency_findings), 1)
        self.assertGreater(latency_findings[0].score, 2000.0)


class TestUnnecessaryRetrieval(unittest.TestCase):
    def test_unnecessary_retrieval_is_detected(self) -> None:
        result = _make_result(
            expected_action="ANSWER",
            action="SEARCH",
            relevant_documents=[],
        )
        findings = diagnose_result(result)
        unnecessary_findings = [
            f for f in findings if f.failure_type == FailureType.UNNECESSARY_RETRIEVAL
        ]
        self.assertEqual(len(unnecessary_findings), 1)


class TestHealthyResult(unittest.TestCase):
    def test_healthy_result_produces_no_findings(self) -> None:
        result = _make_result(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
            latency_ms=100.0,
        )
        findings = diagnose_result(result)
        # With a healthy result, there should be no findings
        self.assertEqual(len(findings), 0)


class TestSeverityCalculation(unittest.TestCase):
    def test_severity_critical_for_large_gap(self) -> None:
        result = _make_result(
            retrieved_documents=["other.md"],
            relevant_documents=["doc1.md", "doc2.md", "doc3.md", "doc4.md", "doc5.md"],
        )
        findings = diagnose_result(result)
        recall_findings = [
            f for f in findings if f.failure_type == FailureType.RETRIEVAL_LOW_RECALL
        ]
        self.assertEqual(len(recall_findings), 1)
        self.assertEqual(recall_findings[0].severity, Severity.CRITICAL)

    def test_severity_low_for_small_gap(self) -> None:
        # Use custom thresholds to create a small gap
        custom_thresholds = dict(DEFAULT_THRESHOLDS)
        custom_thresholds["retrieval_recall"] = 0.65
        result = _make_result(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        # recall = 0.5, threshold = 0.65, gap = 0.15 → HIGH
        findings = diagnose_result(result, thresholds=custom_thresholds)
        recall_findings = [
            f for f in findings if f.failure_type == FailureType.RETRIEVAL_LOW_RECALL
        ]
        self.assertEqual(len(recall_findings), 1)
        self.assertEqual(recall_findings[0].severity, Severity.HIGH)


class TestSystemDiagnosis(unittest.TestCase):
    def test_system_diagnosis_aggregates_multiple_results(self) -> None:
        results = [
            _make_result(
                case_id="c1",
                retrieved_documents=["doc1.md"],
                relevant_documents=["doc1.md", "doc2.md"],
            ),
            _make_result(
                case_id="c2",
                retrieved_documents=["doc1.md"],
                relevant_documents=["doc1.md", "doc2.md"],
            ),
        ]
        diagnosis = diagnose_system(results)
        self.assertEqual(diagnosis.total_cases, 2)
        self.assertEqual(diagnosis.successful_cases, 2)
        self.assertEqual(diagnosis.failed_cases, 0)
        self.assertGreater(len(diagnosis.failure_counts), 0)

    def test_failure_modes_are_ranked(self) -> None:
        results = [
            _make_result(
                case_id="c1",
                retrieved_documents=["other.md"],
                relevant_documents=["doc1.md", "doc2.md"],
            ),
            _make_result(
                case_id="c2",
                retrieved_documents=["other.md"],
                relevant_documents=["doc1.md", "doc2.md"],
            ),
            _make_result(
                case_id="c3",
                answer="The capital of France is Paris.",
                expected_answer="Machine learning is a field of study.",
            ),
        ]
        diagnosis = diagnose_system(results)
        self.assertGreater(len(diagnosis.top_failure_modes), 0)
        # Verify sorted by count descending
        counts = [fm["count"] for fm in diagnosis.top_failure_modes]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_system_diagnosis_empty_results(self) -> None:
        diagnosis = diagnose_system([])
        self.assertEqual(diagnosis.total_cases, 0)
        self.assertEqual(diagnosis.successful_cases, 0)
        self.assertEqual(diagnosis.failed_cases, 0)
        self.assertEqual(len(diagnosis.top_failure_modes), 0)


class TestRootCauseHeuristics(unittest.TestCase):
    def test_retrieval_bottleneck_heuristic(self) -> None:
        """Low recall + low correctness → retrieval bottleneck."""
        results = [
            _make_result(
                case_id="c1",
                retrieved_documents=["other.md"],
                relevant_documents=["doc1.md", "doc2.md", "doc3.md", "doc4.md", "doc5.md"],
                answer="The capital of France is Paris.",
                expected_answer="Machine learning is a field of study.",
            ),
        ]
        diagnosis = diagnose_system(results)
        retrieval_recs = [
            r for r in diagnosis.recommendations if "bottleneck" in r.lower()
        ]
        self.assertGreater(len(retrieval_recs), 0)

    def test_generation_bottleneck_heuristic(self) -> None:
        """High recall + low correctness + low grounding → generation bottleneck."""
        results = [
            _make_result(
                case_id="c1",
                retrieved_documents=["doc1.md"],
                relevant_documents=["doc1.md"],
                answer="The capital of France is Paris.",
                expected_answer="Machine learning is a field of study.",
                retrieved_context="The weather is sunny today.",
            ),
        ]
        diagnosis = diagnose_system(results)
        generation_recs = [
            r for r in diagnosis.recommendations if "generation" in r.lower()
        ]
        self.assertGreater(len(generation_recs), 0)


class TestDiagnosticFindingSerialization(unittest.TestCase):
    def test_finding_to_dict(self) -> None:
        result = _make_result(
            retrieved_documents=["other.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        findings = diagnose_result(result)
        self.assertGreater(len(findings), 0)
        d = findings[0].to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("failure_type", d)
        self.assertIn("severity", d)
        self.assertIn("score", d)
        self.assertIn("threshold", d)
        self.assertIn("message", d)
        self.assertIn("evidence", d)
        self.assertIn("recommendation", d)

    def test_system_diagnosis_to_dict(self) -> None:
        results = [_make_result(case_id="c1")]
        diagnosis = diagnose_system(results)
        d = diagnosis.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("total_cases", d)
        self.assertIn("successful_cases", d)
        self.assertIn("failed_cases", d)
        self.assertIn("failure_counts", d)
        self.assertIn("average_scores", d)
        self.assertIn("top_failure_modes", d)
        self.assertIn("recommendations", d)


class TestConfigurableThresholds(unittest.TestCase):
    def test_custom_thresholds_are_used(self) -> None:
        custom = dict(DEFAULT_THRESHOLDS)
        custom["retrieval_recall"] = 0.95
        result = _make_result(
            retrieved_documents=["doc1.md"],
            relevant_documents=["doc1.md", "doc2.md"],
        )
        # recall = 0.5, default threshold = 0.70 → finding
        default_findings = diagnose_result(result)
        # recall = 0.5, custom threshold = 0.95 → still finding
        custom_findings = diagnose_result(result, thresholds=custom)
        self.assertGreater(len(custom_findings), 0)
        self.assertGreater(len(default_findings), 0)

    def test_default_thresholds_match_expected(self) -> None:
        self.assertEqual(DEFAULT_THRESHOLDS["retrieval_recall"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["retrieval_precision"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["context_relevance"], 0.60)
        self.assertEqual(DEFAULT_THRESHOLDS["answer_correctness"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["answer_relevance"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["answer_grounding"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["answer_completeness"], 0.70)
        self.assertEqual(DEFAULT_THRESHOLDS["decision_score"], 1.0)
        self.assertEqual(DEFAULT_THRESHOLDS["latency_ms"], 2000.0)


if __name__ == "__main__":
    unittest.main()
