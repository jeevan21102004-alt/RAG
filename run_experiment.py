"""CLI demo for the experiment framework.

This script runs a MOCK experiment with 2-3 configurations and prints
a comparison.  No Gemini/API calls are made.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

from src.adaptive_rag.evaluation_schema import EvaluationCase
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_comparison import compare_experiments
from src.adaptive_rag.experiment_runner import run_experiment
from src.adaptive_rag.system_adapter import MockSystemAdapter


# ---------------------------------------------------------------------------
# Mock evaluation cases
# ---------------------------------------------------------------------------

MOCK_CASES: list[EvaluationCase] = [
    EvaluationCase(
        case_id="ml-001",
        question="What is machine learning?",
        category="retrieval_required",
        expected_answer="Machine learning is a field of study about algorithms.",
        expected_action="SEARCH",
        relevant_documents=["machine_learning_intro.md"],
    ),
    EvaluationCase(
        case_id="sl-001",
        question="What is supervised learning?",
        category="retrieval_required",
        expected_answer="Supervised learning uses labeled data.",
        expected_action="SEARCH",
        relevant_documents=["supervised_learning.md"],
    ),
    EvaluationCase(
        case_id="math-001",
        question="What is 2 + 2?",
        category="retrieval_not_required",
        expected_answer="Four.",
        expected_action="ANSWER",
        relevant_documents=[],
    ),
    EvaluationCase(
        case_id="geo-001",
        question="What is the capital of France?",
        category="retrieval_not_required",
        expected_answer="Paris.",
        expected_action="ANSWER",
        relevant_documents=[],
    ),
]


def main() -> None:
    print("=" * 60)
    print("MOCK EXPERIMENT — NO API CALLS MADE")
    print("=" * 60)
    print()

    configs = [
        ExperimentConfig(
            experiment_id="exp-a",
            name="Config A (top_k=3)",
            description="Mock experiment with top_k=3",
            parameters={"top_k": 3, "chunk_size": 300, "retrieval_strategy": "vector"},
            tags=["mock", "top_k_3"],
        ),
        ExperimentConfig(
            experiment_id="exp-b",
            name="Config B (top_k=5)",
            description="Mock experiment with top_k=5",
            parameters={"top_k": 5, "chunk_size": 300, "retrieval_strategy": "vector"},
            tags=["mock", "top_k_5"],
        ),
        ExperimentConfig(
            experiment_id="exp-c",
            name="Config C (top_k=8)",
            description="Mock experiment with top_k=8",
            parameters={"top_k": 8, "chunk_size": 500, "retrieval_strategy": "vector"},
            tags=["mock", "top_k_8"],
        ),
    ]

    adapter = MockSystemAdapter()
    results = []

    for config in configs:
        print(f"Running {config.name}...")
        result = run_experiment(config, MOCK_CASES, adapter)
        results.append(result)
        print(f"  Cases: {result.total_cases}, Success: {result.successful_cases}")
        print()

    comparison = compare_experiments(results)

    print("=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print()

    for result in results:
        print(f"{result.config.name}")
        print(f"  Overall Score:   {result.average_overall_score:.4f}" if result.average_overall_score is not None else "  Overall Score:   N/A")
        print(f"  Answer Score:    {result.average_answer_score:.4f}" if result.average_answer_score is not None else "  Answer Score:    N/A")
        print(f"  Retrieval Score: {result.average_retrieval_score:.4f}" if result.average_retrieval_score is not None else "  Retrieval Score: N/A")
        print(f"  Latency:         {result.average_latency_ms:.1f} ms" if result.average_latency_ms is not None else "  Latency:         N/A")
        print()

    print("-" * 60)
    print(f"BEST OVERALL:     {comparison.best_by_overall_score}")
    print(f"BEST ANSWER:      {comparison.best_by_answer_score}")
    print(f"BEST RETRIEVAL:   {comparison.best_by_retrieval_score}")
    print(f"FASTEST:          {comparison.fastest_experiment}")
    print()
    print("RANKING (by overall score):")
    for i, exp_id in enumerate(comparison.ranking, 1):
        result = next(r for r in results if r.experiment_id == exp_id)
        score = result.average_overall_score
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {i}. {result.config.name} — {score_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
