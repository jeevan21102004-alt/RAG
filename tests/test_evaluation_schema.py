import json
import unittest

from src.adaptive_rag.evaluation_schema import EvaluationCase, EvaluationResult, SystemResponse


class TestEvaluationCase(unittest.TestCase):
    def test_evaluation_case_can_be_created(self) -> None:
        case = EvaluationCase(
            case_id="case-001",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )
        self.assertEqual(case.case_id, "case-001")
        self.assertEqual(case.question, "What is machine learning?")
        self.assertEqual(case.category, "retrieval_required")
        self.assertEqual(case.expected_answer, "Machine learning is a field of study.")
        self.assertEqual(case.expected_action, "SEARCH")
        self.assertEqual(case.relevant_documents, ["machine_learning_intro.md"])


class TestSystemResponse(unittest.TestCase):
    def test_system_response_can_be_created(self) -> None:
        response = SystemResponse(
            answer="Machine learning is a field of study.",
            action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieved_context="Machine learning is a field of study.",
            latency_ms=123.4,
            retrieval_attempts=1,
            metadata={"model": "test-model"},
        )
        self.assertEqual(response.answer, "Machine learning is a field of study.")
        self.assertEqual(response.action, "SEARCH")
        self.assertEqual(response.retrieved_documents, ["machine_learning_intro.md"])
        self.assertEqual(response.retrieved_context, "Machine learning is a field of study.")
        self.assertEqual(response.latency_ms, 123.4)
        self.assertEqual(response.retrieval_attempts, 1)
        self.assertEqual(response.metadata, {"model": "test-model"})


class TestEvaluationResult(unittest.TestCase):
    def test_successful_evaluation_result(self) -> None:
        result = EvaluationResult(
            case_id="case-001",
            question="What is machine learning?",
            expected_answer="Machine learning is a field of study.",
            actual_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            actual_action="SEARCH",
            retrieved_documents=["machine_learning_intro.md"],
            retrieval_attempts=1,
            latency_ms=100.0,
            answer_score=1.0,
            retrieval_score=1.0,
            decision_score=1.0,
            overall_score=1.0,
            status="SUCCESS",
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.overall_score, 1.0)
        self.assertIsNone(result.failure_reason)

    def test_api_error_evaluation_result(self) -> None:
        result = EvaluationResult(
            case_id="case-002",
            question="What is 2 + 2?",
            status="API_ERROR",
            failure_reason="Rate limit exceeded (429)",
        )
        self.assertEqual(result.status, "API_ERROR")
        self.assertEqual(result.failure_reason, "Rate limit exceeded (429)")
        self.assertIsNone(result.overall_score)


class TestOptionalDefaults(unittest.TestCase):
    def test_optional_fields_have_sensible_defaults(self) -> None:
        case = EvaluationCase(case_id="c1", question="Q?", category="general")
        self.assertIsNone(case.expected_answer)
        self.assertIsNone(case.expected_action)
        self.assertEqual(case.relevant_documents, [])

        response = SystemResponse()
        self.assertIsNone(response.answer)
        self.assertIsNone(response.action)
        self.assertEqual(response.retrieved_documents, [])
        self.assertIsNone(response.retrieved_context)
        self.assertIsNone(response.latency_ms)
        self.assertEqual(response.retrieval_attempts, 0)
        self.assertEqual(response.metadata, {})

        result = EvaluationResult(case_id="c1", question="Q?")
        self.assertIsNone(result.expected_answer)
        self.assertIsNone(result.actual_answer)
        self.assertEqual(result.retrieved_documents, [])
        self.assertEqual(result.retrieval_attempts, 0)
        self.assertIsNone(result.latency_ms)
        self.assertIsNone(result.answer_score)
        self.assertIsNone(result.retrieval_score)
        self.assertIsNone(result.decision_score)
        self.assertIsNone(result.overall_score)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIsNone(result.failure_reason)


class TestSerialization(unittest.TestCase):
    def test_objects_serialize_to_json_compatible_dicts(self) -> None:
        case = EvaluationCase(
            case_id="case-001",
            question="What is machine learning?",
            category="retrieval_required",
            expected_action="SEARCH",
        )
        case_dict = case.to_dict()
        self.assertIsInstance(case_dict, dict)
        json.dumps(case_dict)  # must be JSON-serializable

        response = SystemResponse(answer="test", action="ANSWER")
        response_dict = response.to_dict()
        self.assertIsInstance(response_dict, dict)
        json.dumps(response_dict)

        result = EvaluationResult(case_id="case-001", question="Q?", status="SUCCESS")
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        json.dumps(result_dict)


if __name__ == "__main__":
    unittest.main()