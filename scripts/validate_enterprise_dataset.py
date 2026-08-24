"""Validate the enterprise retrieval benchmark dataset.

Checks:
- all referenced documents exist (searched recursively under the KB root)
- all case IDs are unique
- every question has an expected_answer
- every question has at least one relevant document
- expected_action is SEARCH for all questions
- there are no duplicate questions
- document count and question count are reported

Exits with a non-zero status if validation fails.

Usage:
    python scripts/validate_enterprise_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_kb_documents(kb_dir: Path = KB_DIR) -> dict[str, Path]:
    """Map document filename -> path, searching recursively."""
    if not kb_dir.exists():
        return {}
    return {p.name: p for p in sorted(kb_dir.rglob("*.md"))}


def validate(
    questions: list[dict],
    documents: dict[str, Path],
) -> tuple[list[str], int, int]:
    """Return (errors, doc_count, question_count)."""
    errors: list[str] = []

    # Unique case IDs
    ids = [q.get("case_id") for q in questions]
    duplicate_ids = {i for i in ids if ids.count(i) > 1}
    if duplicate_ids:
        errors.append(f"Duplicate case_ids: {sorted(duplicate_ids)}")

    # No duplicate questions
    texts = [q.get("question", "") for q in questions]
    duplicate_questions = {t for t in texts if texts.count(t) > 1}
    if duplicate_questions:
        errors.append(f"Duplicate questions: {sorted(duplicate_questions)}")

    for q in questions:
        cid = q.get("case_id", "<missing-id>")

        if not q.get("question"):
            errors.append(f"{cid}: missing question text")

        if not q.get("expected_answer"):
            errors.append(f"{cid}: missing expected_answer")

        relevant = q.get("relevant_documents", [])
        if not relevant:
            errors.append(f"{cid}: no relevant_documents listed")

        for doc in relevant:
            if doc not in documents:
                errors.append(f"{cid}: referenced document does not exist: {doc}")

        if q.get("expected_action") != "SEARCH":
            errors.append(
                f"{cid}: expected_action must be SEARCH "
                f"(got {q.get('expected_action')!r})"
            )

    return errors, len(documents), len(questions)


def main() -> int:
    questions = load_questions()
    documents = find_kb_documents()

    errors, doc_count, question_count = validate(questions, documents)

    print("=" * 50)
    print("ENTERPRISE DATASET VALIDATION")
    print("=" * 50)
    print(f"Documents found:   {doc_count}")
    print(f"Questions loaded:  {question_count}")
    multi = sum(1 for q in questions if len(q.get("relevant_documents", [])) > 1)
    print(f"Multi-doc questions: {multi}")
    print()

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())