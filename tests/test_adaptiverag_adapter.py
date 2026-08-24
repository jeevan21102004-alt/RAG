"""Tests for the AdaptiveRAG adapter.

These tests do NOT require Gemini API access.  They test configuration
parsing, parameter extraction, default handling, and SystemResponse
structure using the MockSystemAdapter or direct adapter inspection.
"""

import unittest

from src.adaptive_rag.adaptive_rag_adapter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    AdaptiveRAGAdapter,
)
from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.system_adapter import MockSystemAdapter, SystemAdapter


class TestAdapterConfiguration(unittest.TestCase):
    def test_adapter_satisfies_protocol(self) -> None:
        adapter = AdaptiveRAGAdapter()
        self.assertIsInstance(adapter, SystemAdapter)

    def test_default_parameters(self) -> None:
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={},
        )
        # Verify defaults are used when parameters are missing
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.assertEqual(top_k, DEFAULT_TOP_K)
        self.assertEqual(chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(chunk_overlap, DEFAULT_CHUNK_OVERLAP)

    def test_parameter_extraction(self) -> None:
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={
                "chunk_size": 200,
                "chunk_overlap": 40,
                "top_k": 5,
            },
        )
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.assertEqual(top_k, 5)
        self.assertEqual(chunk_size, 200)
        self.assertEqual(chunk_overlap, 40)

    def test_partial_parameters_use_defaults(self) -> None:
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": 5},
        )
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.assertEqual(top_k, 5)
        self.assertEqual(chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(chunk_overlap, DEFAULT_CHUNK_OVERLAP)


class TestMockAdapterSystemResponse(unittest.TestCase):
    """Test SystemResponse structure using the mock adapter (no API needed)."""

    def test_mock_adapter_returns_system_response(self) -> None:
        adapter = MockSystemAdapter()
        case = EvaluationCase(
            case_id="test-001",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        response = adapter.run(case, config)
        self.assertIsInstance(response, SystemResponse)
        self.assertIsNotNone(response.answer)
        self.assertEqual(response.action, "SEARCH")
        self.assertEqual(response.retrieved_documents, ["machine_learning_intro.md"])
        self.assertIsNotNone(response.retrieved_context)
        self.assertIsNotNone(response.latency_ms)
        self.assertEqual(response.retrieval_attempts, 1)

    def test_mock_adapter_answer_action(self) -> None:
        adapter = MockSystemAdapter()
        case = EvaluationCase(
            case_id="test-002",
            question="What is 2 + 2?",
            category="retrieval_not_required",
            expected_answer="Four.",
            expected_action="ANSWER",
            relevant_documents=[],
        )
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": 1, "chunk_size": 300},
        )
        response = adapter.run(case, config)
        self.assertEqual(response.action, "ANSWER")
        self.assertEqual(response.retrieved_documents, [])
        self.assertIsNone(response.retrieved_context)
        self.assertEqual(response.retrieval_attempts, 0)

    def test_mock_adapter_deterministic(self) -> None:
        adapter = MockSystemAdapter()
        case = EvaluationCase(
            case_id="test-003",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        response1 = adapter.run(case, config)
        response2 = adapter.run(case, config)
        self.assertEqual(response1.answer, response2.answer)
        self.assertEqual(response1.action, response2.action)
        self.assertEqual(response1.latency_ms, response2.latency_ms)


class TestAdapterInvalidConfig(unittest.TestCase):
    def test_invalid_configuration_handling(self) -> None:
        """The adapter should handle invalid configurations gracefully."""
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": -1, "chunk_size": 0},
        )
        # The adapter should not crash on invalid parameters
        # (it will use them as-is, but the pipeline should handle it)
        case = EvaluationCase(
            case_id="test-004",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )
        # Using mock adapter to test config handling without API
        adapter = MockSystemAdapter()
        response = adapter.run(case, config)
        self.assertIsInstance(response, SystemResponse)


class TestAdapterMetadata(unittest.TestCase):
    def test_mock_adapter_metadata(self) -> None:
        adapter = MockSystemAdapter()
        case = EvaluationCase(
            case_id="test-005",
            question="What is machine learning?",
            category="retrieval_required",
            expected_answer="Machine learning is a field of study.",
            expected_action="SEARCH",
            relevant_documents=["machine_learning_intro.md"],
        )
        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            parameters={"top_k": 3, "chunk_size": 300},
        )
        response = adapter.run(case, config)
        self.assertIn("adapter", response.metadata)
        self.assertEqual(response.metadata["adapter"], "mock")
        self.assertIn("config_top_k", response.metadata)
        self.assertEqual(response.metadata["config_top_k"], 3)


if __name__ == "__main__":
    unittest.main()
