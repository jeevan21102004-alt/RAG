"""Concrete SystemAdapter for the real AdaptiveRAG pipeline.

This adapter connects the generic experiment framework to the existing
AdaptiveRAG system.  It reuses the existing modules for document loading,
chunking, embeddings, vector storage, retrieval, and Gemini generation.

The adapter is provider-independent at the interface level — it satisfies
the :class:`SystemAdapter` protocol.  Internally it delegates to the
existing AdaptiveRAG pipeline.

IMPORTANT:
- Does NOT duplicate the RAG pipeline.
- Does NOT expose API keys.
- Handles API errors gracefully.
"""

from __future__ import annotations

import time
from typing import Any

from .agent import ANSWER, SEARCH
from .app import AgenticRunResult, build_context, run_agentic_rag
from .chunking import chunk_documents
from .data_loader import load_documents
from .evaluation_schema import EvaluationCase, SystemResponse
from .experiment_config import ExperimentConfig
from .system_adapter import SystemAdapter
from .vector_store import LocalVectorStore


# Default chunk parameters (matching the existing AdaptiveRAG defaults)
DEFAULT_CHUNK_SIZE = 60
DEFAULT_CHUNK_OVERLAP = 15
DEFAULT_TOP_K = 3


class AdaptiveRAGAdapter(SystemAdapter):
    """Concrete adapter that runs the real AdaptiveRAG pipeline.

    The adapter reads ``chunk_size``, ``chunk_overlap``, and ``top_k``
    from :attr:`ExperimentConfig.parameters`.  If a parameter is not
    provided, the existing AdaptiveRAG default is used.

    The vector store is rebuilt for each configuration to ensure that
    chunk parameters are applied correctly.
    """

    def __init__(self) -> None:
        self._store: LocalVectorStore | None = None
        self._store_config: dict[str, Any] | None = None

    def _get_or_build_store(
        self,
        config: ExperimentConfig,
    ) -> LocalVectorStore:
        """Build or reuse a vector store for the given configuration.

        The store is cached per unique chunk configuration to avoid
        rebuilding it for every question.
        """
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        config_key = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

        if self._store is not None and self._store_config == config_key:
            return self._store

        documents = load_documents()
        if not documents:
            raise RuntimeError("No documents found in the data/ directory.")

        chunks = chunk_documents(
            documents,
            chunk_size_words=chunk_size,
            overlap_words=chunk_overlap,
        )
        self._store = LocalVectorStore.build_from_chunks(chunks)
        self._store_config = config_key
        return self._store

    def run(
        self,
        case: EvaluationCase,
        config: ExperimentConfig,
    ) -> SystemResponse:
        """Execute the AdaptiveRAG pipeline for a single evaluation case.

        Parameters
        ----------
        case:
            The evaluation case to process.
        config:
            The experiment configuration (chunk_size, chunk_overlap, top_k).

        Returns
        -------
        SystemResponse
            The system's response, including answer, action, retrieved
            documents, and metadata.
        """
        top_k = config.parameters.get("top_k", DEFAULT_TOP_K)
        chunk_size = config.parameters.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = config.parameters.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

        start_time = time.perf_counter()

        try:
            store = self._get_or_build_store(config)
            run_result = run_agentic_rag(
                case.question,
                top_k=top_k,
                store=store,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return self._convert_run_result(
                run_result, case, config, elapsed_ms,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
            )

        except Exception as error:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SystemResponse(
                answer=None,
                action=None,
                retrieved_documents=[],
                retrieved_context=None,
                latency_ms=elapsed_ms,
                retrieval_attempts=0,
                metadata={
                    "adapter": "adaptiverag",
                    "config_chunk_size": chunk_size,
                    "config_chunk_overlap": chunk_overlap,
                    "config_top_k": top_k,
                    "failure": str(error),
                },
            )

    def _convert_run_result(
        self,
        run_result: AgenticRunResult,
        case: EvaluationCase,
        config: ExperimentConfig,
        latency_ms: float,
        *,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> SystemResponse:
        """Convert an :class:`AgenticRunResult` to a :class:`SystemResponse`."""
        # Determine the action
        action: str | None = None
        if run_result.initial_decision is not None:
            action = run_result.initial_decision.action

        # Extract retrieved document sources
        retrieved_documents: list[str] = []
        for result in run_result.retrieved_results:
            if result.source not in retrieved_documents:
                retrieved_documents.append(result.source)

        # Build retrieved context
        retrieved_context: str | None = None
        if run_result.retrieved_results:
            retrieved_context = build_context(run_result.retrieved_results)

        # Determine the answer
        answer: str | None = run_result.final_answer

        # Build metadata
        metadata: dict[str, Any] = {
            "adapter": "adaptiverag",
            "config_chunk_size": chunk_size,
            "config_chunk_overlap": chunk_overlap,
            "config_top_k": top_k,
            "retrieval_attempts": run_result.retrieval_attempts,
        }

        if run_result.failure:
            metadata["failure"] = run_result.failure

        if run_result.context_evaluations:
            metadata["context_evaluations"] = [
                {
                    "action": eval_result.action,
                    "reason": eval_result.reason,
                }
                for eval_result in run_result.context_evaluations
            ]

        return SystemResponse(
            answer=answer,
            action=action,
            retrieved_documents=retrieved_documents,
            retrieved_context=retrieved_context,
            latency_ms=latency_ms,
            retrieval_attempts=run_result.retrieval_attempts,
            metadata=metadata,
        )
