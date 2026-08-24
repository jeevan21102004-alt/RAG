"""Baseline heuristic answer-quality evaluator.

This module provides a provider-independent, deterministic evaluator that
scores the quality of an AI system's answer against an expected answer.

IMPORTANT — LIMITATIONS
-----------------------
The metrics in this module are **baseline heuristic metrics**.  They are
computed using simple token-overlap and set-similarity techniques and are
**NOT** equivalent to human evaluation or LLM-as-judge scoring.  They exist
to establish a reproducible evaluation framework that can be iterated on
later.

No external APIs, LLM calls, or network access are required.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .evaluation_schema import EvaluationCase, SystemResponse


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

# Common English stop-words used to filter out low-information tokens.
# Includes function words, question words, and contractions.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in",
        "on", "at", "for", "with", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "them", "his",
        "her", "its", "our", "their", "from", "by", "as", "not", "no", "so",
        "than", "too", "very", "s", "t", "don", "ll", "ve", "re", "m",
        # Question words (function words, not content words)
        "what", "how", "why", "when", "where", "which", "who", "whom",
        "whose", "whether",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """Normalize *text* into a list of lower-case, stop-word-filtered tokens.

    The function is deterministic and makes no external calls.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in _STOP_WORDS]


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Return the Jaccard similarity of two token sets (0.0 – 1.0)."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _coverage_ratio(numerator: set[str], denominator: set[str]) -> float:
    """Return the fraction of *denominator* tokens found in *numerator*.

    Returns 1.0 when the denominator is empty (nothing to cover).
    """
    if not denominator:
        return 1.0
    return len(numerator & denominator) / len(denominator)


# ---------------------------------------------------------------------------
# Weights configuration
# ---------------------------------------------------------------------------

DEFAULT_ANSWER_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "relevance": 0.25,
    "grounding": 0.25,
    "completeness": 0.10,
}


@dataclass(frozen=True)
class AnswerScores:
    """Container for the four individual answer-quality dimensions."""

    correctness_score: float
    relevance_score: float
    grounding_score: float
    completeness_score: float
    answer_score: float
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def _score_correctness(expected_answer: str | None, actual_answer: str | None) -> float:
    """Score correctness via Jaccard similarity of token sets.

    A score of 1.0 means the token sets are identical; 0.0 means no overlap.
    """
    expected_tokens = set(tokenize(expected_answer))
    actual_tokens = set(tokenize(actual_answer))
    return _jaccard_similarity(expected_tokens, actual_tokens)


def _score_relevance(question: str, actual_answer: str | None) -> float:
    """Score relevance as the fraction of question content-words present in the answer.

    Uses coverage ratio: |question_terms ∩ answer_terms| / |question_terms|.
    """
    question_tokens = set(tokenize(question))
    answer_tokens = set(tokenize(actual_answer))
    return _coverage_ratio(answer_tokens, question_tokens)


def _score_grounding(actual_answer: str | None, retrieved_context: str | None) -> float:
    """Score grounding as the fraction of answer content-words found in retrieved context.

    If no retrieved context is available the score is 0.0 (cannot verify grounding).
    """
    if not retrieved_context:
        return 0.0
    answer_tokens = set(tokenize(actual_answer))
    context_tokens = set(tokenize(retrieved_context))
    return _coverage_ratio(context_tokens, answer_tokens)


def _score_completeness(expected_answer: str | None, actual_answer: str | None) -> float:
    """Score completeness as the fraction of expected-answer content-words present in the actual answer.

    If no expected answer is available the score is 0.0.
    """
    if not expected_answer:
        return 0.0
    expected_tokens = set(tokenize(expected_answer))
    actual_tokens = set(tokenize(actual_answer))
    return _coverage_ratio(actual_tokens, expected_tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_answer(
    case: EvaluationCase,
    response: SystemResponse,
    weights: dict[str, float] | None = None,
) -> AnswerScores:
    """Evaluate the answer quality of *response* against *case*.

    Parameters
    ----------
    case:
        The evaluation case providing the expected answer and question.
    response:
        The system response providing the actual answer and retrieved context.
    weights:
        Optional override for the dimension weights.  Must contain the keys
        ``correctness``, ``relevance``, ``grounding`` and ``completeness``.
        Defaults to :data:`DEFAULT_ANSWER_WEIGHTS`.

    Returns
    -------
    AnswerScores
        The four dimension scores plus the weighted ``answer_score``.
    """
    w = weights if weights is not None else DEFAULT_ANSWER_WEIGHTS

    correctness = _score_correctness(case.expected_answer, response.answer)
    relevance = _score_relevance(case.question, response.answer)
    grounding = _score_grounding(response.answer, response.retrieved_context)
    completeness = _score_completeness(case.expected_answer, response.answer)

    answer_score = (
        w["correctness"] * correctness
        + w["relevance"] * relevance
        + w["grounding"] * grounding
        + w["completeness"] * completeness
    )

    return AnswerScores(
        correctness_score=correctness,
        relevance_score=relevance,
        grounding_score=grounding,
        completeness_score=completeness,
        answer_score=answer_score,
        weights=dict(w),
    )
