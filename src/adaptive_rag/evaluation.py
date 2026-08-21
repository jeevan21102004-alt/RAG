from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agent import ANSWER, SEARCH
from .app import build_vector_store, run_agentic_rag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation"
RESULTS_PATH = RESULTS_DIR / "results.json"

QUICK_MODE_SIZE = 6
QUICK_MODE_PER_CATEGORY = 2
QUICK_MODE_CATEGORIES = ("retrieval_required", "retrieval_not_required", "insufficient_context")


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    category: str
    expected_action: str
    expected_answer_available: bool
    relevant_documents: list[str]


@dataclass(frozen=True)
class EvaluationRecord:
    question: str
    category: str
    expected_action: str
    expected_answer_available: bool
    relevant_documents: list[str]
    agent_action: str | None
    retrieval_attempts: int
    answer_produced: bool
    status: str
    failure: str | None
    correct_decision: bool


def load_evaluation_dataset(path: Path | None = None) -> list[EvaluationQuestion]:
    dataset_path = path or DATASET_PATH
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [EvaluationQuestion(**item) for item in payload]


def select_quick_questions(dataset: list[EvaluationQuestion]) -> list[EvaluationQuestion]:
    """Deterministically select 2 questions from each of the 3 categories."""
    selected: list[EvaluationQuestion] = []
    for category in QUICK_MODE_CATEGORIES:
        category_questions = [question for question in dataset if question.category == category]
        selected.extend(category_questions[:QUICK_MODE_PER_CATEGORY])
    return selected


def is_api_error(message: str | None) -> bool:
    if not message:
        return False

    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "429",
            "503",
            "timeout",
            "unavailable",
            "resource_exhausted",
            "rate limit",
            "quota",
        )
    )


def classify_status(failure: str | None) -> str:
    if not failure:
        return "SUCCESS"
    if is_api_error(failure):
        return "API_ERROR"
    return "ERROR"


def _decision_is_correct(agent_action: str | None, expected_action: str) -> bool:
    return agent_action == expected_action


def run_evaluation(
    top_k: int = 3,
    delay_between_questions: float = 5.0,
    quick: bool = False,
) -> tuple[list[EvaluationRecord], dict[str, float | int]]:
    dataset = load_evaluation_dataset()
    if quick:
        dataset = select_quick_questions(dataset)
    store = build_vector_store()
    records: list[EvaluationRecord] = []

    for index, question in enumerate(dataset):
        result = run_agentic_rag(question.question, top_k=top_k, store=store)
        # Add a delay between questions to avoid hitting API rate limits
        if index < len(dataset) - 1:
            time.sleep(delay_between_questions)
        agent_action = result.initial_decision.action if result.initial_decision else None
        answer_produced = result.final_answer is not None
        status = classify_status(result.failure)
        correct_decision = _decision_is_correct(agent_action, question.expected_action) if status == "SUCCESS" else False
        records.append(
            EvaluationRecord(
                question=question.question,
                category=question.category,
                expected_action=question.expected_action,
                expected_answer_available=question.expected_answer_available,
                relevant_documents=question.relevant_documents,
                agent_action=agent_action,
                retrieval_attempts=result.retrieval_attempts,
                answer_produced=answer_produced,
                status=status,
                failure=result.failure,
                correct_decision=correct_decision,
            )
        )

    total = len(records)
    successful_records = [record for record in records if record.status == "SUCCESS"]
    api_errors = sum(1 for record in records if record.status == "API_ERROR")
    successful_runs = len(successful_records)

    def category_accuracy(category_name: str) -> float:
        category_records = [record for record in successful_records if record.category == category_name]
        if not category_records:
            return 0.0
        return sum(1 for record in category_records if record.correct_decision) / len(category_records) * 100.0

    retrieval_required_records = [record for record in successful_records if record.category == "retrieval_required"]
    retrieval_not_required_records = [record for record in successful_records if record.category == "retrieval_not_required"]
    insufficient_context_records = [record for record in successful_records if record.category == "insufficient_context"]

    unnecessary_retrievals = sum(1 for record in retrieval_not_required_records if record.agent_action == SEARCH)
    missed_retrievals = sum(1 for record in retrieval_required_records if record.agent_action == ANSWER)
    average_retrieval_attempts = (
        sum(record.retrieval_attempts for record in successful_records) / successful_runs if successful_runs else 0.0
    )
    retrieval_expected_records = retrieval_required_records + insufficient_context_records
    retrieval_success = sum(1 for record in retrieval_expected_records if record.retrieval_attempts > 0)
    retrieval_success_rate = (
        retrieval_success / len(retrieval_expected_records) * 100.0 if retrieval_expected_records else 0.0
    )

    metrics = {
        "total_questions": total,
        "successful_runs": successful_runs,
        "api_errors": api_errors,
        "overall_decision_accuracy": (
            sum(1 for record in successful_records if record.correct_decision) / successful_runs * 100.0
            if successful_runs
            else 0.0
        ),
        "retrieval_required_accuracy": category_accuracy("retrieval_required"),
        "retrieval_not_required_accuracy": category_accuracy("retrieval_not_required"),
        "insufficient_context_accuracy": category_accuracy("insufficient_context"),
        "unnecessary_retrieval_rate": (
            unnecessary_retrievals / len(retrieval_not_required_records) * 100.0 if retrieval_not_required_records else 0.0
        ),
        "missed_retrieval_rate": (
            missed_retrievals / len(retrieval_required_records) * 100.0 if retrieval_required_records else 0.0
        ),
        "average_retrieval_attempts": average_retrieval_attempts,
        "retrieval_success_rate": retrieval_success_rate,
    }

    return records, metrics


def save_results(
    records: list[EvaluationRecord],
    metrics: dict[str, float | int],
    path: Path | None = None,
    mode: str = "full",
) -> None:
    results_path = path or RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": mode,
        "api_error_count": metrics.get("api_errors", 0),
        "metrics": metrics,
        "records": [asdict(record) for record in records],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_record(index: int, record: EvaluationRecord) -> str:
    decision = "CORRECT" if record.correct_decision else "INCORRECT"
    failure_line = f"Failure: {record.failure}\n" if record.failure else ""
    return (
        f"Question {index}\n"
        f"Category: {record.category}\n"
        f"Expected: {record.expected_action}\n"
        f"Agent: {record.agent_action}\n"
        f"Retrieval attempts: {record.retrieval_attempts}\n"
        f"Answer produced: {record.answer_produced}\n"
        f"Status: {record.status}\n"
        f"Decision: {decision}\n"
        f"{failure_line}"
    )
