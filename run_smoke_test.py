"""Tiny smoke test for the real AdaptiveRAG adapter.

DIAGNOSTIC MODE — runs 1 configuration with at most 2 evaluation
questions and aborts immediately on the first error.  Designed to make
at most ~2 Gemini API calls (one initial decision + one generation per
question, and it stops after the first question succeeds).

Usage:
    python run_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.adaptive_rag_adapter import AdaptiveRAGAdapter
from src.adaptive_rag.evaluation_schema import EvaluationCase
from src.adaptive_rag.experiment_config import ExperimentConfig


SMOKE_QUESTIONS: list[dict] = [
    {
        "question": "What is 2 + 2?",
        "category": "retrieval_not_required",
        "expected_action": "ANSWER",
        "relevant_documents": [],
    },
    {
        "question": "What is machine learning?",
        "category": "retrieval_required",
        "expected_action": "SEARCH",
        "relevant_documents": ["machine_learning_intro.md"],
    },
]


def main() -> int:
    print("=" * 60)
    print("ADAPTIVERAG SMOKE TEST (1 config, max 2 questions)")
    print("=" * 60)
    print()

    config = ExperimentConfig(
        experiment_id="smoke_test",
        name="Smoke Test Config A",
        description="Diagnostic smoke test — chunk=100, overlap=20, top_k=3",
        parameters={"chunk_size": 100, "chunk_overlap": 20, "top_k": 3},
        tags=["smoke", "diagnostic"],
    )

    cases = [
        EvaluationCase(
            case_id=f"smoke-{i}",
            question=item["question"],
            category=item["category"],
            expected_answer=None,
            expected_action=item["expected_action"],
            relevant_documents=item["relevant_documents"],
        )
        for i, item in enumerate(SMOKE_QUESTIONS, start=1)
    ]

    adapter = AdaptiveRAGAdapter()

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Question: {case.question}")
        try:
            response = adapter.run(case, config)
        except Exception:
            print("RESULT: EXCEPTION (unhandled by adapter)")
            traceback.print_exc()
            return 1

        failure = response.metadata.get("failure")
        print(f"  action={response.action!r} attempts={response.retrieval_attempts} "
              f"latency={response.latency_ms:.0f}ms docs={response.retrieved_documents}")
        if response.answer is not None:
            preview = response.answer[:120].replace(chr(10), " ")
            print(f"  answer_preview: {preview}")
        if failure:
            print(f"  FAILURE METADATA: {failure}")
            lowered = failure.lower()
            if any(m in lowered for m in ("429", "quota", "rate limit", "resource_exhausted")):
                category = "API_QUOTA / RATE_LIMIT"
            elif any(m in lowered for m in ("401", "403", "api key", "api_key", "permission")):
                category = "AUTHENTICATION"
            elif any(m in lowered for m in ("timeout", "timed out")):
                category = "TIMEOUT"
            elif any(m in lowered for m in ("503", "unavailable", "internal server")):
                category = "API_UNAVAILABLE"
            else:
                category = "OTHER_RUNTIME_ERROR"
            print(f"  ERROR CATEGORY: {category}")
            print()
            print("SMOKE TEST FAILED — stopping. No retries will be attempted.")
            return 1

        print("  STATUS: OK")
        print()

    print("=" * 60)
    print("SMOKE TEST PASSED — real adapter works end-to-end.")
    print("Stopping here. Full experiment NOT run.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())