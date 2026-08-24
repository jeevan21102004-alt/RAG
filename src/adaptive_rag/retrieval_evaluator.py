"""Deterministic retrieval-quality evaluation.

This module provides provider-independent, deterministic metrics for
evaluating the quality of document retrieval.  No external APIs, LLM calls,
or network access are required.

IMPORTANT — LIMITATIONS
-----------------------
The context-relevance metric in this module is a **baseline heuristic
lexical metric**.  It does NOT represent semantic understanding.  It checks
whether content-words from the question appear in the retrieved context,
which is a weak proxy for true relevance.

"Lexical context relevance is a baseline heuristic and does not represent
semantic relevance."
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .answer_evaluator import tokenize


# ---------------------------------------------------------------------------
# Document ID normalization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_doc_id(doc_id: str) -> str:
    """Normalize a document identifier for consistent comparison.

    Normalization rules:
    1. Convert to lower-case.
    2. Strip leading/trailing whitespace.
    3. Remove common path prefixes (``data/``, ``docs/``, ``./``).
    4. Remove the ``.md`` extension if present.
    5. Extract the core alphanumeric token sequence.

    Examples
    --------
    >>> normalize_doc_id("data/supervised_learning.md")
    'supervised_learning'
    >>> normalize_doc_id("supervised_learning.md")
    'supervised_learning'
    >>> normalize_doc_id("./docs/Machine_Learning.md")
    'machine_learning'

    This is **not** fuzzy matching — two IDs are considered equal only when
    their normalized forms are identical.
    """
    if not doc_id:
        return ""

    normalized = doc_id.lower().strip()

    # Remove common path prefixes
    for prefix in ("data/", "docs/", "./", "../"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    # Remove file extension
    if normalized.endswith(".md"):
        normalized = normalized[:-3]

    # Extract core alphanumeric tokens (handles remaining path separators)
    tokens = _TOKEN_RE.findall(normalized)
    return "_".join(tokens) if tokens else normalized


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalMetrics:
    """Structured result of a retrieval-quality evaluation.

    All scores are in the range [0.0, 1.0].
    """

    precision: float
    recall: float
    f1: float
    retrieved_count: int
    relevant_count: int
    relevant_retrieved_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_document_lists(
    retrieved_documents: list[str],
    relevant_documents: list[str],
) -> tuple[list[str], list[str]]:
    """Normalize both document lists and remove duplicates from retrieved."""
    normalized_relevant = [normalize_doc_id(doc) for doc in relevant_documents]
    normalized_retrieved = [normalize_doc_id(doc) for doc in retrieved_documents]
    # Remove duplicates while preserving order
    seen: set[str] = set()
    deduped_retrieved: list[str] = []
    for doc in normalized_retrieved:
        if doc not in seen:
            seen.add(doc)
            deduped_retrieved.append(doc)
    return deduped_retrieved, normalized_relevant


def calculate_retrieval_metrics(
    retrieved_documents: list[str],
    relevant_documents: list[str],
) -> RetrievalMetrics:
    """Calculate precision, recall, and F1 for a retrieval result.

    Parameters
    ----------
    retrieved_documents:
        Document identifiers returned by the system (may contain duplicates
        or path-prefixed variants).
    relevant_documents:
        Ground-truth document identifiers that should have been retrieved.

    Returns
    -------
    RetrievalMetrics
        A dataclass containing precision, recall, f1, and counts.

    Edge-case behavior
    ------------------
    * **No relevant documents** — precision is 0.0 (no relevant docs to
      retrieve), recall is 0.0, F1 is 0.0.
    * **No retrieved documents** — precision is 0.0, recall is 0.0, F1 is
      0.0.
    * **Perfect retrieval** — precision = recall = F1 = 1.0.
    * **Duplicate retrieved documents** — duplicates are removed before
      calculation.
    """
    retrieved, relevant = _normalize_document_lists(
        retrieved_documents, relevant_documents
    )

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    relevant_retrieved = retrieved_set & relevant_set
    relevant_retrieved_count = len(relevant_retrieved)

    retrieved_count = len(retrieved_set)
    relevant_count = len(relevant_set)

    # Precision: relevant retrieved / all retrieved
    if retrieved_count == 0:
        precision = 0.0
    else:
        precision = relevant_retrieved_count / retrieved_count

    # Recall: relevant retrieved / all relevant
    if relevant_count == 0:
        recall = 0.0
    else:
        recall = relevant_retrieved_count / relevant_count

    # F1: harmonic mean of precision and recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        retrieved_count=retrieved_count,
        relevant_count=relevant_count,
        relevant_retrieved_count=relevant_retrieved_count,
    )


def calculate_retrieval_metrics_at_k(
    retrieved_documents: list[str],
    relevant_documents: list[str],
    k: int,
) -> RetrievalMetrics:
    """Calculate retrieval metrics considering only the top-*k* retrieved documents.

    Parameters
    ----------
    retrieved_documents:
        Document identifiers returned by the system, ordered by rank.
    relevant_documents:
        Ground-truth document identifiers.
    k:
        The number of top-ranked documents to consider.

    Returns
    -------
    RetrievalMetrics
        Metrics computed over the first *k* retrieved documents.

    If *k* exceeds the number of retrieved documents, all retrieved
    documents are used.
    """
    top_k = retrieved_documents[:k] if k > 0 else []
    return calculate_retrieval_metrics(top_k, relevant_documents)


# ---------------------------------------------------------------------------
# Context relevance
# ---------------------------------------------------------------------------

def calculate_context_relevance(
    question: str,
    retrieved_context: str | None,
) -> float:
    """Calculate a heuristic lexical context-relevance score.

    This metric measures what fraction of the question's content-words
    (non-stop-words) appear in the retrieved context.

    .. warning::
        This is a **baseline heuristic lexical metric** and does NOT
        represent semantic relevance.  It only checks for term overlap.

    Parameters
    ----------
    question:
        The user's question.
    retrieved_context:
        The text retrieved from the document store.

    Returns
    -------
    float
        A score in [0.0, 1.0].  Returns 0.0 if the context is empty or
        None.  Returns 1.0 if the question has no content-words.
    """
    if not retrieved_context:
        return 0.0

    question_tokens = set(tokenize(question))
    context_tokens = set(tokenize(retrieved_context))

    if not question_tokens:
        return 1.0

    return len(question_tokens & context_tokens) / len(question_tokens)
