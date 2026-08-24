"""Controlled retrieval benchmark adapter (forced SEARCH).

This adapter evaluates RETRIEVAL CONFIGURATIONS independently of the
Agentic ANSWER/SEARCH decision layer.  It ALWAYS performs retrieval and
NEVER asks the agent whether retrieval is needed.

    Question → FORCED SEARCH → Retriever → Configuration → Evaluation

The production agent is NOT modified — this adapter reuses the existing
data loader, chunker, embeddings, vector store, and retrieval modules.

Latency is measured separately:
- ``retrieval_latency_ms``: time spent in the retriever.
- ``generation_latency_ms``: time spent in LLM answer generation (if enabled).
- ``total_latency_ms``: sum of the two.

No API keys are exposed.  Generation can be disabled for API-free testing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .app import build_context
from .chunking import chunk_documents
from .data_loader import load_documents
from .evaluation_schema import EvaluationCase, SystemResponse
from .models import Document
from .experiment_config import ExperimentConfig
from .llm import generate_answer
from .retrieval import retrieve
from .system_adapter import SystemAdapter
from .vector_store import LocalVectorStore


# Default chunk parameters (matching the existing AdaptiveRAG defaults)
DEFAULT_CHUNK_SIZE = 60
DEFAULT_CHUNK_OVERLAP = 15
DEFAULT_TOP_K = 3


def _load_documents_recursive(folder: Path) -> list[Document]:
    """Load markdown/text documents recursively from *folder*.

    Mirrors :func:`data_loader.load_documents` but descends into
    subdirectories (e.g., ``enterprise_kb/hr/*.md``).
    """
    paths = sorted([*folder.rglob("*.md"), *folder.rglob("*.txt")])
    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents


class RetrievalBenchmarkAdapter(SystemAdapter):
    """Forced-retrieval adapter for controlled configuration benchmarks.

    The adapter reads ``chunk_size``, ``chunk_overlap``, and ``top_k``
    from :attr:`ExperimentConfig.parameters`.  If a parameter is not
    provided, the existing AdaptiveRAG default is used.

    Parameters
    ----------
    enable_generation:
        When ``True`` (default), a final answer is generated with the
        existing Gemini generation code so that answer quality can also be
        measured.  When ``False``, no LLM call is made — useful for
        API-free unit tests and pure retrieval benchmarking.
    data_dir:
        Optional corpus directory.  When provided, documents are loaded
        recursively from this directory instead of the default
        ``data/`` folder (e.g., ``data/enterprise_kb``).  The normal
        AdaptiveRAG data directory is never modified.
    """

    def __init__(
        self,
        enable_generation: bool = True,
        data_dir: Path | None = None,
    ) -> None:
        self.enable_generation = enable_generation
        self.data_dir = data_dir
        self._store: LocalVectorStore | None = None
        self._store_config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Vector store management
    # ------------------------------------------------------------------

    def _get_or_build_store(self, config: ExperimentConfig) -> LocalVectorStore:
        """Build or reuse a vector store for the given chunk configuration."""
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        config_key = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "data_dir": str(self.data_dir) if self.data_dir else None,
        }

        if self._store is not None and self._store_config == config_key:
            return self._store

        if self.data_dir is not None:
            documents = _load_documents_recursive(self.data_dir)
        else:
            documents = load_documents()
        if not documents:
            raise RuntimeError("No documents found in the data directory.")

        chunks = chunk_documents(
            documents,
            chunk_size_words=chunk_size,
            overlap_words=chunk_overlap,
        )
        self._store = LocalVectorStore.build_from_chunks(chunks)
        self._store_config = config_key
        return self._store

    # ------------------------------------------------------------------
    # Forced-retrieval execution
    # ------------------------------------------------------------------

    def run(
        self,
        case: EvaluationCase,
        config: ExperimentConfig,
    ) -> SystemResponse:
        """Always retrieve, then optionally generate an answer.

        The agent decision layer is intentionally bypassed: SEARCH is
        forced so that retrieval quality is directly measurable.
        """
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        metadata: dict[str, Any] = {
            "adapter": "retrieval_benchmark",
            "forced_search": True,
            "config_chunk_size": chunk_size,
            "config_chunk_overlap": chunk_overlap,
            "config_top_k": top_k,
        }

        try:
            store = self._get_or_build_store(config)

            # --- Retrieval (latency measured separately) ---------------
            retrieval_start = time.perf_counter()
            results = retrieve(case.question, store, top_k=top_k)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

            retrieved_documents: list[str] = []
            for result in results:
                if result.source not in retrieved_documents:
                    retrieved_documents.append(result.source)

            retrieved_context = build_context(results) if results else ""

            metadata["retrieval_latency_ms"] = round(retrieval_latency_ms, 3)
            metadata["chunks_retrieved"] = len(results)

            # --- Optional generation (latency measured separately) -----
            generation_latency_ms: float | None = None
            answer: str | None = None
            failure: str | None = None

            if self.enable_generation:
                generation_start = time.perf_counter()
                try:
                    answer = generate_answer(case.question, retrieved_context)
                except Exception as error:
                    failure = f"LLM generation failed: {error}"
                finally:
                    generation_latency_ms = (
                        time.perf_counter() - generation_start
                    ) * 1000.0

            metadata["generation_latency_ms"] = (
                round(generation_latency_ms, 3)
                if generation_latency_ms is not None
                else None
            )
            total_latency_ms = retrieval_latency_ms + (
                generation_latency_ms or 0.0
            )
            metadata["total_latency_ms"] = round(total_latency_ms, 3)

            if failure:
                metadata["failure"] = failure

            return SystemResponse(
                answer=answer,
                action="SEARCH",
                retrieved_documents=retrieved_documents,
                retrieved_context=retrieved_context or None,
                latency_ms=total_latency_ms,
                retrieval_attempts=1,
                metadata=metadata,
            )

        except Exception as error:
            # Record the failure; never fabricate metrics.
            metadata["failure"] = str(error)
            return SystemResponse(
                answer=None,
                action="SEARCH",
                retrieved_documents=[],
                retrieved_context=None,
                latency_ms=None,
                retrieval_attempts=0,
                metadata=metadata,
            )