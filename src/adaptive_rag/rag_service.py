"""Practical RAG service (Phase 7A).

Thin, Streamlit-independent service layer around the already-validated
retrieval configuration (chunk_size=300, overlap=40, top_k=2).

Reuses existing AdaptiveRAG components only -- no duplicated retrieval
logic, no LLM calls, no external API:

- document loading: ``retrieval_benchmark_adapter._load_documents_recursive``
- chunking: ``chunking.chunk_documents``
- vector store: ``vector_store.LocalVectorStore``
- retrieval: ``retrieval.retrieve``
- context building: ``app.build_context``

Corpus: ``data/enterprise_kb/`` (local, deterministic).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"

# Validated configuration (Phases 6C/6E evidence: fixed top_k=2 dominates).
VALIDATED_CHUNK_SIZE = 300
VALIDATED_CHUNK_OVERLAP = 40
VALIDATED_TOP_K = 2


@dataclass(frozen=True)
class RAGResponse:
    """Structured retrieval-only response for one user query."""

    query: str
    context: str
    sources: list[str] = field(default_factory=list)
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


def get_validated_config() -> dict[str, int]:
    """Return the validated retrieval configuration."""
    return {
        "chunk_size": VALIDATED_CHUNK_SIZE,
        "chunk_overlap": VALIDATED_CHUNK_OVERLAP,
        "top_k": VALIDATED_TOP_K,
    }


def _validate_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string")
    return cleaned


@lru_cache(maxsize=1)
def get_vector_store():  # type: ignore[no-untyped-def]
    """Build (once) the validated vector store over the enterprise corpus."""
    from .chunking import chunk_documents
    from .retrieval_benchmark_adapter import _load_documents_recursive
    from .vector_store import LocalVectorStore

    if not ENTERPRISE_KB_DIR.exists():
        raise RuntimeError(
            f"Enterprise knowledge base not found: {ENTERPRISE_KB_DIR}"
        )
    documents = _load_documents_recursive(ENTERPRISE_KB_DIR)
    if not documents:
        raise RuntimeError("No documents found in the enterprise knowledge base.")
    chunks = chunk_documents(
        documents,
        chunk_size_words=VALIDATED_CHUNK_SIZE,
        overlap_words=VALIDATED_CHUNK_OVERLAP,
    )
    store = LocalVectorStore.build_from_chunks(chunks)
    return store


def query_rag(query: str, top_k: int = VALIDATED_TOP_K) -> RAGResponse:
    """Retrieve top_k chunks for *query* and build a structured response.

    Retrieval-only: never calls an LLM or any external API.
    """
    from .retrieval import retrieve

    cleaned = _validate_query(query)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    store = get_vector_store()
    started = time.perf_counter()
    results = retrieve(cleaned, store, top_k=top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0

    retrieved_documents = [
        {
            "source": item.source,
            "chunk_index": item.chunk_index,
            "text": item.text,
            "score": item.score,
            "document_id": f"{item.source}#chunk-{item.chunk_index}",
        }
        for item in results
    ]
    sources: list[str] = []
    for doc in retrieved_documents:
        if doc["source"] not in sources:
            sources.append(doc["source"])

    # Reuse the existing context builder for a consistent format.
    from .app import build_context

    context = build_context(results) if results else ""

    metadata: dict[str, Any] = {
        "chunk_size": VALIDATED_CHUNK_SIZE,
        "chunk_overlap": VALIDATED_CHUNK_OVERLAP,
        "top_k_requested": top_k,
        "num_returned": len(retrieved_documents),
        "num_documents": len({d.source for d in store.records}),
        "num_chunks": len(store.records),
        "retrieval_latency_ms": latency_ms,
        "generation_used": False,
    }
    return RAGResponse(
        query=cleaned,
        context=context,
        sources=sources,
        retrieved_documents=retrieved_documents,
        retrieval_metadata=metadata,
    )


def clear_cache() -> None:
    """Clear the cached vector store (mainly for tests)."""
    get_vector_store.cache_clear()
