"""Deterministic AI failure diagnosis engine.

This module analyzes :class:`EvaluationResult` objects and identifies
likely failure modes of AI systems.  It is provider-independent and makes
**no API calls**.

IMPORTANT — LIMITATIONS
-----------------------
The diagnostic heuristics in this module are **deterministic rules** based
on threshold comparisons.  They do **NOT** prove causal relationships.
They are designed to:

1. Surface likely failure modes for human investigation.
2. Provide actionable recommendations for debugging.
3. Enable systematic analysis of evaluation results.

They are **not** designed to:
- Replace root-cause analysis by a human expert.
- Guarantee causal conclusions.
- Replace LLM-based or statistical diagnostic methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .evaluation_schema import EvaluationResult


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    """Types of AI-system failure modes that can be diagnosed."""

    RETRIEVAL_LOW_RECALL = "RETRIEVAL_LOW_RECALL"
    RETRIEVAL_LOW_PRECISION = "RETRIEVAL_LOW_PRECISION"
    CONTEXT_LOW_RELEVANCE = "CONTEXT_LOW_RELEVANCE"
    ANSWER_LOW_CORRECTNESS = "ANSWER_LOW_CORRECTNESS"
    ANSWER_LOW_RELEVANCE = "ANSWER_LOW_RELEVANCE"
    ANSWER_LOW_GROUNDING = "ANSWER_LOW_GROUNDING"
    ANSWER_LOW_COMPLETENESS = "ANSWER_LOW_COMPLETENESS"
    DECISION_ERROR = "DECISION_ERROR"
    HIGH_LATENCY = "HIGH_LATENCY"
    UNNECESSARY_RETRIEVAL = "UNNECESSARY_RETRIEVAL"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    """Severity levels for diagnostic findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "retrieval_recall": 0.70,
    "retrieval_precision": 0.70,
    "context_relevance": 0.60,
    "answer_correctness": 0.70,
    "answer_relevance": 0.70,
    "answer_grounding": 0.70,
    "answer_completeness": 0.70,
    "decision_score": 1.0,
    "latency_ms": 2000.0,
}


# ---------------------------------------------------------------------------
# Diagnostic finding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticFinding:
    """A single diagnostic finding from analyzing an EvaluationResult.

    Attributes
    ----------
    failure_type:
        The type of failure detected.
    severity:
        How severe the finding is (LOW, MEDIUM, HIGH, CRITICAL).
    score:
        The actual metric value that triggered the finding.
    threshold:
        The threshold that was violated.
    message:
        Human-readable description of what happened.
    evidence:
        Supporting evidence for the finding.
    recommendation:
        Suggested next steps for investigation.
    """

    failure_type: FailureType
    severity: Severity
    score: float | None
    threshold: float
    message: str
    evidence: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "message": self.message,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Severity calculation
# ---------------------------------------------------------------------------

def _severity_below_threshold(
    score: float,
    threshold: float,
    critical_gap: float = 0.30,
    high_gap: float = 0.15,
    medium_gap: float = 0.05,
) -> Severity:
    """Determine severity based on how far below threshold a score is.

    Severity rules (for scores below threshold):
    - Gap >= 0.30 → CRITICAL
    - Gap >= 0.15 → HIGH
    - Gap >= 0.05 → MEDIUM
    - Gap < 0.05 → LOW

    Parameters
    ----------
    score:
        The actual metric value.
    threshold:
        The threshold that was violated.
    critical_gap, high_gap, medium_gap:
        Gap thresholds for severity levels.
    """
    gap = threshold - score
    if gap >= critical_gap:
        return Severity.CRITICAL
    if gap >= high_gap:
        return Severity.HIGH
    if gap >= medium_gap:
        return Severity.MEDIUM
    return Severity.LOW


def _severity_above_threshold(
    score: float,
    threshold: float,
    critical_gap: float = 500.0,
    high_gap: float = 1000.0,
    medium_gap: float = 500.0,
) -> Severity:
    """Determine severity for metrics that exceed a threshold (e.g., latency).

    Severity rules (for scores above threshold):
    - Excess >= 1000 → CRITICAL
    - Excess >= 500 → HIGH
    - Excess >= 250 → MEDIUM
    - Excess < 250 → LOW
    """
    excess = score - threshold
    if excess >= high_gap:
        return Severity.CRITICAL
    if excess >= critical_gap:
        return Severity.HIGH
    if excess >= medium_gap:
        return Severity.MEDIUM
    return Severity.LOW


# ---------------------------------------------------------------------------
# Individual diagnostic checks
# ---------------------------------------------------------------------------

def _check_retrieval_recall(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low retrieval recall."""
    metrics = result.metadata.get("retrieval_metrics")
    if not metrics:
        return None

    recall = metrics.get("recall")
    if recall is None:
        return None

    threshold = thresholds["retrieval_recall"]
    if recall >= threshold:
        return None

    severity = _severity_below_threshold(recall, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.RETRIEVAL_LOW_RECALL,
        severity=severity,
        score=recall,
        threshold=threshold,
        message="Retriever returned too few relevant documents.",
        evidence=f"Recall = {recall:.4f}",
        recommendation=(
            "Investigate chunking strategy, embedding model, top-k, "
            "or retrieval algorithm."
        ),
    )


def _check_retrieval_precision(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low retrieval precision."""
    metrics = result.metadata.get("retrieval_metrics")
    if not metrics:
        return None

    precision = metrics.get("precision")
    if precision is None:
        return None

    threshold = thresholds["retrieval_precision"]
    if precision >= threshold:
        return None

    severity = _severity_below_threshold(precision, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.RETRIEVAL_LOW_PRECISION,
        severity=severity,
        score=precision,
        threshold=threshold,
        message="Retriever returned too many irrelevant documents.",
        evidence=f"Precision = {precision:.4f}",
        recommendation=(
            "Investigate retrieval ranking, query expansion, or "
            "filtering strategy."
        ),
    )


def _check_context_relevance(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low context relevance."""
    context_relevance = result.metadata.get("context_relevance")
    if context_relevance is None:
        return None

    threshold = thresholds["context_relevance"]
    if context_relevance >= threshold:
        return None

    severity = _severity_below_threshold(context_relevance, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.CONTEXT_LOW_RELEVANCE,
        severity=severity,
        score=context_relevance,
        threshold=threshold,
        message="Retrieved context does not sufficiently address the question.",
        evidence=f"Context relevance = {context_relevance:.4f}",
        recommendation=(
            "Investigate retrieval query formulation, document "
            "selection, or context window size."
        ),
    )


def _check_answer_correctness(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low answer correctness."""
    answer_scores = result.metadata.get("answer_scores")
    if not answer_scores:
        return None

    correctness = answer_scores.get("correctness_score")
    if correctness is None:
        return None

    threshold = thresholds["answer_correctness"]
    if correctness >= threshold:
        return None

    severity = _severity_below_threshold(correctness, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.ANSWER_LOW_CORRECTNESS,
        severity=severity,
        score=correctness,
        threshold=threshold,
        message="Answer does not match the expected answer.",
        evidence=f"Correctness = {correctness:.4f}",
        recommendation=(
            "Investigate answer generation, prompt design, or "
            "retrieval quality."
        ),
    )


def _check_answer_relevance(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low answer relevance."""
    answer_scores = result.metadata.get("answer_scores")
    if not answer_scores:
        return None

    relevance = answer_scores.get("relevance_score")
    if relevance is None:
        return None

    threshold = thresholds["answer_relevance"]
    if relevance >= threshold:
        return None

    severity = _severity_below_threshold(relevance, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.ANSWER_LOW_RELEVANCE,
        severity=severity,
        score=relevance,
        threshold=threshold,
        message="Answer does not address the question.",
        evidence=f"Relevance = {relevance:.4f}",
        recommendation=(
            "Investigate whether the answer addresses the question "
            "or if the system is off-topic."
        ),
    )


def _check_answer_grounding(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low answer grounding."""
    answer_scores = result.metadata.get("answer_scores")
    if not answer_scores:
        return None

    grounding = answer_scores.get("grounding_score")
    if grounding is None:
        return None

    threshold = thresholds["answer_grounding"]
    if grounding >= threshold:
        return None

    severity = _severity_below_threshold(grounding, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.ANSWER_LOW_GROUNDING,
        severity=severity,
        score=grounding,
        threshold=threshold,
        message="Answer contains claims not supported by retrieved context.",
        evidence=f"Grounding = {grounding:.4f}",
        recommendation=(
            "Investigate whether the answer is hallucinated or "
            "whether retrieved context is insufficient."
        ),
    )


def _check_answer_completeness(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for low answer completeness."""
    answer_scores = result.metadata.get("answer_scores")
    if not answer_scores:
        return None

    completeness = answer_scores.get("completeness_score")
    if completeness is None:
        return None

    threshold = thresholds["answer_completeness"]
    if completeness >= threshold:
        return None

    severity = _severity_below_threshold(completeness, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.ANSWER_LOW_COMPLETENESS,
        severity=severity,
        score=completeness,
        threshold=threshold,
        message="Answer is missing key information from the expected answer.",
        evidence=f"Completeness = {completeness:.4f}",
        recommendation=(
            "Investigate whether the answer is too brief or "
            "omits important details."
        ),
    )


def _check_decision_error(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for decision errors (expected vs. actual action mismatch)."""
    decision_score = result.decision_score
    if decision_score is None:
        return None

    threshold = thresholds["decision_score"]
    if decision_score >= threshold:
        return None

    return DiagnosticFinding(
        failure_type=FailureType.DECISION_ERROR,
        severity=Severity.HIGH,
        score=decision_score,
        threshold=threshold,
        message="System action does not match expected action.",
        evidence=(
            f"Expected action = {result.expected_action}, "
            f"Actual action = {result.actual_action}"
        ),
        recommendation=(
            "Investigate the retrieval policy or decision-making "
            "logic."
        ),
    )


def _check_high_latency(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for high latency."""
    latency = result.latency_ms
    if latency is None:
        return None

    threshold = thresholds["latency_ms"]
    if latency <= threshold:
        return None

    severity = _severity_above_threshold(latency, threshold)
    return DiagnosticFinding(
        failure_type=FailureType.HIGH_LATENCY,
        severity=severity,
        score=latency,
        threshold=threshold,
        message="Response time exceeds acceptable threshold.",
        evidence=f"Latency = {latency:.1f} ms",
        recommendation=(
            "Investigate model inference time, retrieval speed, "
            "or system bottlenecks."
        ),
    )


def _check_unnecessary_retrieval(
    result: EvaluationResult,
    thresholds: dict[str, float],
) -> DiagnosticFinding | None:
    """Check for unnecessary retrieval (expected ANSWER but got SEARCH)."""
    expected_action = result.expected_action
    actual_action = result.actual_action

    if expected_action is None or actual_action is None:
        return None

    if expected_action == "ANSWER" and actual_action == "SEARCH":
        return DiagnosticFinding(
            failure_type=FailureType.UNNECESSARY_RETRIEVAL,
            severity=Severity.MEDIUM,
            score=None,
            threshold=0.0,
            message="System performed retrieval when direct answer was expected.",
            evidence=(
                f"Expected action = {expected_action}, "
                f"Actual action = {actual_action}"
            ),
            recommendation=(
                "Investigate whether the retrieval policy is too "
                "aggressive for simple questions."
            ),
        )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnose_result(
    result: EvaluationResult,
    thresholds: dict[str, float] | None = None,
) -> list[DiagnosticFinding]:
    """Diagnose a single :class:`EvaluationResult` and return findings.

    Parameters
    ----------
    result:
        The evaluation result to diagnose.
    thresholds:
        Optional override for the diagnostic thresholds.  Defaults to
        :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    list[DiagnosticFinding]
        A list of findings.  An empty list means no issues were detected.
    """
    t = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

    checks = [
        _check_retrieval_recall,
        _check_retrieval_precision,
        _check_context_relevance,
        _check_answer_correctness,
        _check_answer_relevance,
        _check_answer_grounding,
        _check_answer_completeness,
        _check_decision_error,
        _check_high_latency,
        _check_unnecessary_retrieval,
    ]

    findings: list[DiagnosticFinding] = []
    for check in checks:
        finding = check(result, t)
        if finding is not None:
            findings.append(finding)

    return findings


@dataclass(frozen=True)
class SystemDiagnosis:
    """Aggregated diagnosis across multiple evaluation results.

    Attributes
    ----------
    total_cases:
        Total number of evaluation results analyzed.
    successful_cases:
        Number of results with status SUCCESS.
    failed_cases:
        Number of results with a non-SUCCESS status.
    failure_counts:
        Mapping of FailureType to count across all results.
    average_scores:
        Average of each available score metric across all results.
    top_failure_modes:
        List of (FailureType, count, percentage) sorted by frequency.
    recommendations:
        Aggregated recommendations from root-cause heuristics.
    """

    total_cases: int
    successful_cases: int
    failed_cases: int
    failure_counts: dict[str, int] = field(default_factory=dict)
    average_scores: dict[str, float] = field(default_factory=dict)
    top_failure_modes: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _root_cause_heuristics(
    failure_counts: dict[str, int],
    average_scores: dict[str, float],
) -> list[str]:
    """Generate root-cause heuristic recommendations.

    These are **deterministic heuristics**, not guaranteed causal
    conclusions.
    """
    recommendations: list[str] = []

    recall = average_scores.get("retrieval_recall")
    precision = average_scores.get("retrieval_precision")
    correctness = average_scores.get("correctness_score")
    grounding = average_scores.get("grounding_score")
    context_rel = average_scores.get("context_relevance")
    decision_errors = failure_counts.get(
        FailureType.DECISION_ERROR.value, 0
    )

    # Heuristic 1: Retrieval bottleneck
    if recall is not None and recall < 0.70 and correctness is not None and correctness < 0.70:
        recommendations.append(
            "Retrieval may be the primary bottleneck: low recall combined "
            "with low answer correctness suggests the retriever is missing "
            "relevant documents."
        )

    # Heuristic 2: Generation bottleneck
    if (
        recall is not None and recall >= 0.70
        and correctness is not None and correctness < 0.70
        and grounding is not None and grounding < 0.70
    ):
        recommendations.append(
            "Generation may not be using retrieved evidence effectively: "
            "high recall but low correctness and low grounding suggest the "
            "generator is not leveraging retrieved context."
        )

    # Heuristic 3: Excessive irrelevant context
    if (
        precision is not None and precision < 0.70
        and context_rel is not None and context_rel < 0.60
    ):
        recommendations.append(
            "Retriever may be returning excessive irrelevant context: "
            "low precision combined with low context relevance."
        )

    # Heuristic 4: Decision policy issues
    if decision_errors > 0:
        recommendations.append(
            "Retrieval policy may require optimization: frequent decision "
            "errors detected."
        )

    return recommendations


def diagnose_system(
    results: list[EvaluationResult],
    thresholds: dict[str, float] | None = None,
) -> SystemDiagnosis:
    """Aggregate diagnostics across multiple evaluation results.

    Parameters
    ----------
    results:
        List of evaluation results to analyze.
    thresholds:
        Optional override for the diagnostic thresholds.

    Returns
    -------
    SystemDiagnosis
        Aggregated diagnosis with failure counts, average scores,
        top failure modes, and root-cause recommendations.
    """
    t = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.status == "SUCCESS")
    failed_cases = total_cases - successful_cases

    failure_counts: dict[str, int] = {}
    score_accumulators: dict[str, list[float]] = {}

    for result in results:
        findings = diagnose_result(result, t)
        for finding in findings:
            failure_counts[finding.failure_type.value] = (
                failure_counts.get(finding.failure_type.value, 0) + 1
            )

        # Collect scores for averaging
        metrics = result.metadata.get("retrieval_metrics")
        if metrics:
            for key in ("precision", "recall", "f1"):
                val = metrics.get(key)
                if val is not None:
                    score_accumulators.setdefault("retrieval_" + key, []).append(val)

        context_rel = result.metadata.get("context_relevance")
        if context_rel is not None:
            score_accumulators.setdefault("context_relevance", []).append(context_rel)

        answer_scores = result.metadata.get("answer_scores")
        if answer_scores:
            for key in ("correctness_score", "relevance_score", "grounding_score", "completeness_score"):
                val = answer_scores.get(key)
                if val is not None:
                    score_accumulators.setdefault(key, []).append(val)

        if result.decision_score is not None:
            score_accumulators.setdefault("decision_score", []).append(result.decision_score)

        if result.latency_ms is not None:
            score_accumulators.setdefault("latency_ms", []).append(result.latency_ms)

    # Calculate averages
    average_scores: dict[str, float] = {}
    for key, values in score_accumulators.items():
        if values:
            average_scores[key] = sum(values) / len(values)

    # Rank failure modes by frequency
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    top_failure_modes: list[dict[str, Any]] = []
    for failure_type, count in sorted_failures:
        percentage = (count / total_cases * 100.0) if total_cases > 0 else 0.0
        top_failure_modes.append({
            "failure_type": failure_type,
            "count": count,
            "percentage": round(percentage, 1),
        })

    # Root-cause heuristics
    recommendations = _root_cause_heuristics(failure_counts, average_scores)

    return SystemDiagnosis(
        total_cases=total_cases,
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        failure_counts=failure_counts,
        average_scores=average_scores,
        top_failure_modes=top_failure_modes,
        recommendations=recommendations,
    )
