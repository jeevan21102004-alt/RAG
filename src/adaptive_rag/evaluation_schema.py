from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    """A single evaluation case describing an input and its expected outcome.

    This is the data contract for what an evaluation engine evaluates.
    It is independent of any specific AI system or API.
    """

    case_id: str
    question: str
    category: str
    expected_answer: str | None = None
    expected_action: str | None = None
    relevant_documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SystemResponse:
    """The raw response produced by an AI system under evaluation.

    This represents the output of any AI system (RAG, LLM app, etc.)
    before any evaluation logic is applied.
    """

    answer: str | None = None
    action: str | None = None
    retrieved_documents: list[str] = field(default_factory=list)
    retrieved_context: str | None = None
    latency_ms: float | None = None
    retrieval_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of evaluating a SystemResponse against an EvaluationCase.

    Scores are optional and may be filled in by the evaluation engine.
    status indicates the outcome: SUCCESS, API_ERROR, ERROR, etc.
    """

    case_id: str
    question: str
    expected_answer: str | None = None
    actual_answer: str | None = None
    expected_action: str | None = None
    actual_action: str | None = None
    retrieved_documents: list[str] = field(default_factory=list)
    retrieval_attempts: int = 0
    latency_ms: float | None = None
    answer_score: float | None = None
    retrieval_score: float | None = None
    decision_score: float | None = None
    overall_score: float | None = None
    status: str = "SUCCESS"
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)