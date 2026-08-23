from __future__ import annotations

import re
from dataclasses import dataclass


# Keywords that suggest the question refers to the provided documents.
DOCUMENT_KEYWORDS = (
    "according",
    "document",
    "provided",
    "mentioned",
    "sample",
    "described",
    "describe",
    "recommended",
    "used in the sample",
    "in the documents",
    "in the sample",
)

# Question-type keywords that often indicate retrieval is needed.
RETRIEVAL_HINT_KEYWORDS = (
    "what",
    "why",
    "how",
    "compare",
    "explain",
    "which",
    "where",
    "when",
)


@dataclass(frozen=True)
class StateFeatures:
    question_length: int
    word_count: int
    question_mark_count: int
    keyword_hits: int
    document_reference: int


def _count_keyword_hits(question: str) -> int:
    lowered = question.lower()
    return sum(1 for keyword in RETRIEVAL_HINT_KEYWORDS if keyword in lowered)


def _has_document_reference(question: str) -> int:
    lowered = question.lower()
    return 1 if any(keyword in lowered for keyword in DOCUMENT_KEYWORDS) else 0


def question_to_state(question: str) -> list[float]:
    """Convert a question into a deterministic numerical feature vector.

    This is a pure function and does NOT use an LLM or embeddings.
    """
    lowered = question.lower()
    words = re.findall(r"[a-z0-9']+", lowered)

    features = StateFeatures(
        question_length=len(question),
        word_count=len(words),
        question_mark_count=question.count("?"),
        keyword_hits=_count_keyword_hits(question),
        document_reference=_has_document_reference(question),
    )

    return [
        float(features.question_length),
        float(features.word_count),
        float(features.question_mark_count),
        float(features.keyword_hits),
        float(features.document_reference),
    ]


def state_feature_names() -> list[str]:
    """Return the names of the features in the state vector, in order."""
    return [
        "question_length",
        "word_count",
        "question_mark_count",
        "keyword_hits",
        "document_reference",
    ]
