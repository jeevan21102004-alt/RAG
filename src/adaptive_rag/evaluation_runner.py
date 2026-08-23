"""Evaluation runner that produces an :class:`EvaluationResult`.

This module ties together the answer-quality evaluator, the decision scorer
and the retrieval scorer to produce a single :class:`EvaluationResult` from
an :class:`EvaluationCase` and a :class:`SystemResponse`.

The runner is **provider-independent** and makes **no API calls**.  All
scoring is deterministic and heuristic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .answer_evaluator import AnswerScores, DEFAULT_ANSWER_WEIGHTS, evaluate_answer
from .evaluation_schema import EvaluationCase, EvaluationResult, SystemResponse


# ---------------------------------------------------------------------------
# Overall-score weights
# ---------------------------------------------------------------------------

DEFAULT_OVERALL_WEIGHTS: dict[str, float] = {
    "answer": 0.60,
    "retrieval": 0.25,
    "decision": 0.15,
}


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------

def calculate_retrieval_score(
    retrieved_documents: list[str],
    relevant_documents: list[str],
) -> float | None:
    """Calculate a simple recall-based retrieval score.

    ``retrieval_score = |relevant ∩ retrieved| / |relevant|``

    Edge-case behaviour
    -------------------
    * **No relevant documents** — returns ``None`` (retrieval cannot be
      evaluated; there is nothing to retrieve).
    * **No retrieved documents** — returns ``0.0`` (everything was missed).
    * **Perfect retrieval** — returns ``1.0``.
    * **Partial retrieval** — returns the fraction of relevant documents
      that were retrieved.

    Parameters
    ----------
    retrieved_documents:
        Document identifiers returned by the system.
    relevant_documents:
        Ground-truth document identifiers that should have been retrieved.

    Returns
    -------
    float | None
        A score in [0.0, 1.0], or ``None`` when there are no relevant
        documents to compare against.
    """
    if not relevant_documents:
        return None

    relevant_set = set(relevant_documents)
    retrieved_set = set(retrieved_documents)

    if not retrieved_set:
        return 0.0

    relevant_retrieved = relevant_set & retrieved_set
    return len(relevant_retrieved) / len(relevant_set)


# ---------------------------------------------------------------------------
# Decision scoring
# ---------------------------------------------------------------------------

def calculate_decision_score(
    expected_action: str | None,
    actual_action: str | None,
) -> float | None:
    """Calculate a binary decision score.

    * Returns ``1.0`` when *expected_action* and *actual_action* are both
      present and equal.
    * Returns ``0.0`` when both are present but differ.
    * Returns ``None`` when either value is missing (cannot evaluate).
    """
    if expected_action is None or actual_action is None:
        return None
    return 1.0 if expected_action == actual_action else 0.0


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------

def calculate_overall_score(
    answer_score: float | None,
    retrieval_score: float | None,
    decision_score: float | None,
    weights: dict[str, float] | None = None,
) -> float | None:
    """Calculate the overall score with proportional weight redistribution.

    Default weights:
        * answer:   0.60
        * retrieval: 0.25
        * decision:  0.15

    If a component is ``None`` (unavailable), its weight is redistributed
    **proportionally** among the remaining available components.  If *all*
    components are ``None``, the overall score is ``None``.

    Parameters
    ----------
    answer_score, retrieval_score, decision_score:
        Individual component scores (each may be ``None``).
    weights:
        Optional override for the component weights.

    Returns
    -------
    float | None
        The weighted overall score in [0.0, 1.0], or ``None`` when no
        component scores are available.
    """
    w = weights if weights is not None else DEFAULT_OVERALL_WEIGHTS

    components: list[tuple[float | None, float]] = [
        (answer_score, w["answer"]),
        (retrieval_score, w["retrieval"]),
        (decision_score, w["decision"]),
    ]

    available = [(score, weight) for score, weight in components if score is not None]

    if not available:
        return None

    total_weight = sum(weight for _, weight in available)
    weighted_sum = sum(score * weight for score, weight in available)
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationScores:
    """Structured container for all component scores of an evaluation."""

    answer_scores: AnswerScores | None
    retrieval_score: float | None
    decision_score: float | None
    overall_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_evaluation(
    case: EvaluationCase,
    response: SystemResponse,
    answer_weights: dict[str, float] | None = None,
    overall_weights: dict[str, float] | None = None,
) -> EvaluationResult:
    """Run a full evaluation and return an :class:`EvaluationResult`.

    Steps
    -----
    1. Evaluate answer quality (correctness, relevance, grounding,
       completeness → ``answer_score``).
    2. Calculate the decision score (if expected/actual actions exist).
    3. Calculate the retrieval score (if relevant documents exist).
    4. Calculate the overall score with proportional weight
       redistribution.
    5. Assemble and return an :class:`EvaluationResult`.

    The individual component scores are stored in the result's ``metadata``
    field so that the existing :class:`EvaluationResult` schema is not
    modified.
    """
    # --- 1. Answer quality ------------------------------------------------
    answer_scores: AnswerScores | None = None
    answer_score: float | None = None
    if response.answer is not None:
        answer_scores = evaluate_answer(case, response, weights=answer_weights)
        answer_score = answer_scores.answer_score

    # --- 2. Decision score ------------------------------------------------
    decision_score = calculate_decision_score(
        case.expected_action, response.action
    )

    # --- 3. Retrieval score -----------------------------------------------
    retrieval_score = calculate_retrieval_score(
        response.retrieved_documents, case.relevant_documents
    )

    # --- 4. Overall score -------------------------------------------------
    overall_score = calculate_overall_score(
        answer_score, retrieval_score, decision_score, weights=overall_weights
    )

    # --- 5. Assemble result ------------------------------------------------
    metadata: dict[str, Any] = dict(response.metadata)

    if answer_scores is not None:
        metadata["answer_scores"] = answer_scores.to_dict()

    metadata["retrieval_score"] = retrieval_score
    metadata["decision_score"] = decision_score
    metadata["overall_weights"] = (
        overall_weights if overall_weights is not None else DEFAULT_OVERALL_WEIGHTS
    )

    return EvaluationResult(
        case_id=case.case_id,
        question=case.question,
        expected_answer=case.expected_answer,
        actual_answer=response.answer,
        expected_action=case.expected_action,
        actual_action=response.action,
        retrieved_documents=response.retrieved_documents,
        retrieval_attempts=response.retrieval_attempts,
        latency_ms=response.latency_ms,
        answer_score=answer_score,
        retrieval_score=retrieval_score,
        decision_score=decision_score,
        overall_score=overall_score,
        status="SUCCESS",
        failure_reason=None,
        metadata=metadata,
    )
