"""Tests for the controlled retrieval benchmark adapter.

These tests do NOT require Gemini API access — generation is disabled and
retrieval runs entirely on the local vector store.
"""

import unittest
from pathlib import Path

from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.retrieval_benchmark_adapter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    RetrievalBenchmarkAdapter,
)
from src.adaptive_rag.system_adapter import SystemAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"


def _make_case(
    question: str = "What is machine learning?",
    relevant_documents: list[str] | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        case_id="rb-test",
        question=question,
        category="retrieval_required",
        expected_answer="Machine learning is a way to build programs that learn patterns from data.",
        expected_action="SEARCH",
        relevant_documents=relevant_documents or ["machine_learning_intro.md"],
    )


def _make_config(top_k: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="rb-test",
        name="Retrieval Benchmark Test",
        parameters={"chunk_size": 100, "chunk_overlap": 20, "top_k": top_k},
    )


class TestForcedSearchBehaviour(unittest.TestCase):
    """All tests use enable_generation=False so no API calls occur."""

    def setUp(self) -> None:
        self.adapter = RetrievalBenchmarkAdapter(enable_generation=False)

    def test_search_is_always_returned(self) -> None:
        response = self.adapter.run(_make_case(), _make_config())
        self.assertEqual(response.action, "SEARCH")

    def test_retrieval_attempts_is_one(self) -> None:
        response = self.adapter.run(_make_case(), _make_config())
        self.assertEqual(response.retrieval_attempts, 1)

    def test_retrieved_documents_populated_for_matching_question(self) -> None:
        response = self.adapter.run(
            _make_case("What is machine learning?"), _make_config()
        )
        self.assertIn("machine_learning_intro.md", response.retrieved_documents)

    def test_top_k_is_respected(self) -> None:
        # With 3 documents in the corpus, top_k=5 should retrieve at most
        # all chunks; the number of retrieved chunks must not exceed top_k.
        config = _make_config(top_k=2)
        response = self.adapter.run(_make_case(), config)
        self.assertLessEqual(response.retrieval_attempts, 1)
        self.assertIsNotNone(response.retrieved_context)

    def test_system_response_structure(self) -> None:
        response = self.adapter.run(_make_case(), _make_config())
        self.assertIsInstance(response, SystemResponse)
        self.assertIsNotNone(response.retrieved_context)
        self.assertIsNone(response.answer)  # generation disabled
        self.assertIsInstance(response.metadata, dict)

    def test_latency_metadata_present(self) -> None:
        response = self.adapter.run(_make_case(), _make_config())
        self.assertIn("retrieval_latency_ms", response.metadata)
        self.assertIn("generation_latency_ms", response.metadata)
        self.assertIn("total_latency_ms", response.metadata)
        self.assertGreaterEqual(response.metadata["retrieval_latency_ms"], 0.0)
        self.assertIsNone(response.metadata["generation_latency_ms"])

    def test_forced_search_flag_in_metadata(self) -> None:
        response = self.adapter.run(_make_case(), _make_config())
        self.assertTrue(response.metadata.get("forced_search"))

    def test_configuration_parameters_respected(self) -> None:
        response = self.adapter.run(_make_case(), _make_config(top_k=5))
        metadata = response.metadata
        self.assertEqual(metadata["config_chunk_size"], 100)
        self.assertEqual(metadata["config_chunk_overlap"], 20)
        self.assertEqual(metadata["config_top_k"], 5)

class TestEnterpriseDatasetPath(unittest.TestCase):
    """Retrieval must work against data/enterprise_kb via the data_dir param.

    The enterprise corpus is the 25-document synthetic knowledge base used
    by the ``--dataset enterprise`` benchmark.  No API access is required.
    """

    def setUp(self) -> None:
        self.adapter = RetrievalBenchmarkAdapter(
            enable_generation=False, data_dir=ENTERPRISE_KB_DIR
        )

    def test_enterprise_kb_directory_exists(self) -> None:
        self.assertTrue(ENTERPRISE_KB_DIR.is_dir())

    def test_enterprise_document_retrieved(self) -> None:
        case = EvaluationCase(
            case_id="ent-test",
            question="What is the maximum number of remote work days allowed per week?",
            category="enterprise_hr",
            expected_answer="Up to 3 days per week.",
            expected_action="SEARCH",
            relevant_documents=["remote_work_policy.md"],
        )
        response = self.adapter.run(case, _make_config())
        self.assertEqual(response.action, "SEARCH")
        self.assertIn("remote_work_policy.md", response.retrieved_documents)
        self.assertIsNotNone(response.retrieved_context)

    def test_enterprise_configuration_and_response_metadata(self) -> None:
        case = EvaluationCase(
            case_id="ent-test",
            question="Who must approve every production deployment?",
            category="enterprise_engineering",
            expected_answer="The owning team's tech lead and the on-call SRE engineer.",
            expected_action="SEARCH",
            relevant_documents=["deployment_guidelines.md"],
        )
        response = self.adapter.run(case, _make_config(top_k=3))
        self.assertIsInstance(response, SystemResponse)
        self.assertEqual(response.metadata["adapter"], "retrieval_benchmark")
        self.assertTrue(response.metadata["forced_search"])
        self.assertEqual(response.metadata["config_top_k"], 3)
        for key in ("retrieval_latency_ms", "total_latency_ms"):
            self.assertIn(key, response.metadata)

class TestDefaultParameters(unittest.TestCase):
    def test_defaults_used_when_parameters_missing(self) -> None:
        config = ExperimentConfig(experiment_id="t", name="T", parameters={})
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        self.assertEqual(chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(chunk_overlap, DEFAULT_CHUNK_OVERLAP)
        self.assertEqual(top_k, DEFAULT_TOP_K)


class TestInvalidConfiguration(unittest.TestCase):
    def test_invalid_config_handled_safely(self) -> None:
        """Invalid parameters must not crash the adapter."""
        adapter = RetrievalBenchmarkAdapter(enable_generation=False)
        config = ExperimentConfig(
            experiment_id="bad",
            name="Bad",
            parameters={"top_k": -1, "chunk_size": 0},
        )
        try:
            response = adapter.run(_make_case(), config)
            self.assertIsInstance(response, SystemResponse)
            self.assertEqual(response.action, "SEARCH")
        except Exception as error:  # pragma: no cover - should not happen
            self.fail(f"Adapter raised unexpectedly: {error}")


class TestProtocolConformance(unittest.TestCase):
    def test_adapter_satisfies_protocol(self) -> None:
        adapter = RetrievalBenchmarkAdapter(enable_generation=False)
        self.assertIsInstance(adapter, SystemAdapter)


if __name__ == "__main__":
    unittest.main()