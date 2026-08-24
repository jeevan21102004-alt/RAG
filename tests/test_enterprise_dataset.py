"""Tests for the enterprise retrieval benchmark dataset.

These tests validate the synthetic enterprise knowledge base and the
benchmark question set.  No API access is required.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_enterprise_dataset import (  # noqa: E402
    find_kb_documents,
    load_questions,
    validate,
)

KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"


class TestEnterpriseDocuments(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = find_kb_documents(KB_DIR)

    def test_documents_exist(self) -> None:
        self.assertGreaterEqual(len(self.documents), 20)

    def test_expected_document_count(self) -> None:
        self.assertEqual(len(self.documents), 25)

    def test_all_categories_present(self) -> None:
        categories = {p.parent.name for p in self.documents.values()}
        for expected in ("hr", "engineering", "finance", "product", "security"):
            self.assertIn(expected, categories)

    def test_documents_are_non_empty(self) -> None:
        for name, path in self.documents.items():
            text = path.read_text(encoding="utf-8").strip()
            self.assertGreater(len(text.split()), 100, f"{name} too short")


class TestEnterpriseQuestions(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = load_questions(QUESTIONS_PATH)
        self.documents = find_kb_documents(KB_DIR)

    def test_questions_load(self) -> None:
        self.assertGreaterEqual(len(self.questions), 20)

    def test_ids_are_unique(self) -> None:
        ids = [q["case_id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_expected_actions_are_search(self) -> None:
        for q in self.questions:
            self.assertEqual(q["expected_action"], "SEARCH", q["case_id"])

    def test_relevant_documents_exist(self) -> None:
        for q in self.questions:
            self.assertTrue(q.get("relevant_documents"), q["case_id"])
            for doc in q["relevant_documents"]:
                self.assertIn(doc, self.documents, f"{q['case_id']}: {doc}")

    def test_every_question_has_expected_answer(self) -> None:
        for q in self.questions:
            self.assertTrue(q.get("expected_answer"), q["case_id"])


class TestDatasetValidationUtility(unittest.TestCase):
    def test_validation_passes_on_real_dataset(self) -> None:
        questions = load_questions(QUESTIONS_PATH)
        documents = find_kb_documents(KB_DIR)
        errors, doc_count, question_count = validate(questions, documents)
        self.assertEqual(errors, [])
        self.assertEqual(doc_count, 25)
        self.assertEqual(question_count, 20)

    def test_validation_detects_missing_document(self) -> None:
        questions = [
            {
                "case_id": "x1",
                "question": "Q?",
                "expected_answer": "A",
                "expected_action": "SEARCH",
                "relevant_documents": ["nonexistent.md"],
            }
        ]
        errors, _, _ = validate(questions, {})
        self.assertTrue(any("does not exist" in e for e in errors))

    def test_validation_detects_duplicate_ids(self) -> None:
        base = {
            "question": "Q?",
            "expected_answer": "A",
            "expected_action": "SEARCH",
            "relevant_documents": [],
        }
        questions = [
            {"case_id": "dup", **base},
            {"case_id": "dup", **base},
        ]
        errors, _, _ = validate(questions, {})
        self.assertTrue(any("Duplicate case_ids" in e for e in errors))

    def test_validation_rejects_non_search_action(self) -> None:
        questions = [
            {
                "case_id": "x2",
                "question": "Q?",
                "expected_answer": "A",
                "expected_action": "ANSWER",
                "relevant_documents": [],
            }
        ]
        errors, _, _ = validate(questions, {})
        self.assertTrue(any("must be SEARCH" in e for e in errors))


if __name__ == "__main__":
    unittest.main()