"""CLI for running real AdaptiveRAG experiments.

This script:
1. Loads 4 configurations from experiments/adaptiverag_configs.json.
2. Loads the evaluation dataset from data/evaluation_questions.json.
3. Converts each configuration to ExperimentConfig.
4. Runs the real AdaptiveRAG adapter.
5. Evaluates each response using the existing evaluation engine.
6. Runs the existing diagnosis engine.
7. Saves each ExperimentResult as JSON (incrementally, after each config).
8. Compares all experiments.
9. Prints a final comparison table.

IMPORTANT:
- This script makes Gemini API calls (the real AdaptiveRAG system
  generates answers).
- No API keys are printed.
- If API quota is exhausted: the error is recorded, the partial result is
  saved, and execution continues to the next configuration.
- No retries beyond the pipeline's built-in behaviour; no model switching.
- All values come from actual evaluation results — nothing is fabricated.

Usage:
    python run_adaptiverag_experiments.py                     # full run
    python run_adaptiverag_experiments.py --max-questions 2   # pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.adaptive_rag_adapter import AdaptiveRAGAdapter
from src.adaptive_rag.evaluation_schema import EvaluationCase
from src.adaptive_rag.retrieval_benchmark_adapter import RetrievalBenchmarkAdapter
from src.adaptive_rag.experiment_comparison import compare_experiments
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.experiment_result import ExperimentResult
from src.adaptive_rag.experiment_runner import run_experiment
from src.adaptive_rag.experiment_storage import save_experiment_result


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIGS_PATH = PROJECT_ROOT / "experiments" / "adaptiverag_configs.json"
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_questions.json"
RETRIEVAL_BENCHMARK_DATASET_PATH = (
    PROJECT_ROOT / "data" / "retrieval_benchmark_questions.json"
)
ENTERPRISE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "enterprise_retrieval_questions.json"
)
ENTERPRISE_KB_DIR = PROJECT_ROOT / "data" / "enterprise_kb"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Expected answers for evaluation questions
# ---------------------------------------------------------------------------

# These expected answers are derived from the content of the sample
# documents (data/*.md).  They are NOT a new dataset — they are the
# ground-truth answers for the existing evaluation questions.
EXPECTED_ANSWERS: dict[str, str] = {
    "What is machine learning?": "Machine learning is a field of study that gives computers the ability to learn from data.",
    "What is supervised learning?": "Supervised learning uses labeled data to train models.",
    "What tasks is supervised learning used for?": "Supervised learning is used for classification and regression tasks.",
    "What is overfitting?": "Overfitting occurs when a model learns the training data too well, including noise.",
    "How does the document describe a model that overfits?": "A model that overfits performs well on training data but poorly on unseen data.",
    "What common workflow is described for machine learning?": "The workflow includes data collection, model training, and evaluation.",
    "Which example tasks are mentioned for supervised learning?": "Classification and regression are mentioned as example tasks.",
    "What does the document say machine learning programs learn from?": "Machine learning programs learn from data.",
    "What is 2 + 2?": "Four.",
    "What is the capital of France?": "Paris.",
    "What color is the sky on a clear day?": "Blue.",
    "What comes after Monday?": "Tuesday.",
    "How many days are in a week?": "Seven.",
    "Is water wet?": "Yes.",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_configs() -> list[ExperimentConfig]:
    """Load experiment configurations from JSON."""
    data = json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
    return [ExperimentConfig(**item) for item in data]


def load_evaluation_cases(max_questions: int | None = None) -> list[EvaluationCase]:
    """Load evaluation questions and convert to EvaluationCase objects.

    Selection is deterministic: the first *max_questions* entries of the
    dataset are used when a limit is provided.  The same selection is
    shared by every configuration.
    """
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if max_questions is not None:
        data = data[:max_questions]
    cases: list[EvaluationCase] = []
    for item in data:
        question = item["question"]
        expected_answer = EXPECTED_ANSWERS.get(question)
        cases.append(
            EvaluationCase(
                case_id=f"q-{question[:20].replace(' ', '_')}",
                question=question,
                category=item["category"],
                expected_answer=expected_answer,
                expected_action=item["expected_action"],
                relevant_documents=item.get("relevant_documents", []),
            )
        )
    return cases


def load_retrieval_benchmark_cases(
    max_questions: int | None = None,
    dataset: str = "default",
) -> list[EvaluationCase]:
    """Load retrieval-benchmark questions (forced-SEARCH dataset).

    The dataset already contains case_id, question, category,
    expected_answer, expected_action, and relevant_documents.  Selection
    is deterministic: the first *max_questions* entries.

    ``dataset`` selects the question set: ``default`` uses the original
    8-question benchmark; ``enterprise`` uses the 20-question enterprise
    knowledge-base benchmark.
    """
    path = (
        ENTERPRISE_DATASET_PATH
        if dataset == "enterprise"
        else RETRIEVAL_BENCHMARK_DATASET_PATH
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    if max_questions is not None:
        data = data[:max_questions]
    return [EvaluationCase(**item) for item in data]


def count_api_errors(result: ExperimentResult) -> int:
    """Count API_ERROR records inside an ExperimentResult."""
    records = result.metadata.get("evaluation_results", [])
    return sum(1 for record in records if record.get("status") == "API_ERROR")


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run real AdaptiveRAG experiments.")
    parser.add_argument(
        "--mode",
        choices=("agent", "retrieval"),
        default="agent",
        help="agent: run the full Agentic RAG pipeline (default). "
             "retrieval: controlled benchmark with FORCED SEARCH — the "
             "agent decision layer is bypassed so retrieval configurations "
             "are compared directly.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Deterministically limit to the first N evaluation questions "
             "(same questions used for all configurations).",
    )
    parser.add_argument(
        "--dataset",
        choices=("default", "enterprise"),
        default="default",
        help="Question set for retrieval mode: 'default' (original "
             "3-document corpus) or 'enterprise' (25-document synthetic "
             "enterprise knowledge base). Ignored in agent mode.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between questions and between configurations "
             "(default: 5.0).",
    )
    args = parser.parse_args()

    configs = load_configs()

    if args.mode == "retrieval":
        cases = load_retrieval_benchmark_cases(
            max_questions=args.max_questions,
            dataset=args.dataset,
        )
        data_dir = ENTERPRISE_KB_DIR if args.dataset == "enterprise" else None
        # Pilot API rule: the primary purpose of retrieval mode is the
        # evaluation of retrieval CONFIGURATIONS, not answer generation.
        # The existing evaluation architecture computes all retrieval metrics
        # (precision, recall, F1, context relevance) without an answer, so
        # generation is disabled to keep Gemini usage near zero.
        adapter: AdaptiveRAGAdapter | RetrievalBenchmarkAdapter = (
            RetrievalBenchmarkAdapter(enable_generation=False, data_dir=data_dir)
        )
        comparison_filename = (
            f"adaptiverag_retrieval_{args.dataset}_comparison.json"
        )
    else:
        cases = load_evaluation_cases(max_questions=args.max_questions)
        adapter = AdaptiveRAGAdapter()
        comparison_filename = "adaptiverag_comparison.json"

    mode_label = (
        "CONTROLLED RETRIEVAL BENCHMARK (FORCED SEARCH)"
        if args.mode == "retrieval"
        else "AGENTIC RAG (agent decides SEARCH/ANSWER)"
    )

    if args.max_questions is not None:
        header = (
            f"PHASE 3B PILOT ({mode_label}) — {len(cases)} QUESTIONS PER "
            f"CONFIGURATION ({len(cases)} x {len(configs)} = "
            f"{len(cases) * len(configs)} total)"
        )
    else:
        header = (
            f"ADAPTIVERAG EXPERIMENTS ({mode_label}) — {len(cases)} QUESTIONS "
            f"PER CONFIGURATION"
        )

    print("=" * 60)
    print(header)
    print("=" * 60)
    print()
    print(f"Loaded {len(configs)} configurations and {len(cases)} evaluation cases.")
    print(f"Inter-question / inter-config delay: {args.delay:.1f}s")
    print()

    results: list[ExperimentResult] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for config_index, config in enumerate(configs):
        print(f"[{config_index + 1}/{len(configs)}] Running {config.name} ...")
        try:
            result = run_experiment(
                config,
                cases,
                adapter,
                delay_between_questions=args.delay,
            )
            results.append(result)
            # Incremental save immediately after this configuration finishes.
            save_path = RESULTS_DIR / f"{result.experiment_id}.json"
            save_experiment_result(result, path=save_path)
            api_errors = count_api_errors(result)
            print(
                f"  Cases: {result.total_cases} | Success: {result.successful_cases}"
                f" | Failed: {result.failed_cases} | API errors: {api_errors}"
            )
            print(f"  Overall: {_fmt(result.average_overall_score)}"
                  f" | Avg latency:"
                  f" {result.average_latency_ms:.0f} ms"
                  if result.average_latency_ms is not None else
                  f"  Overall: {_fmt(result.average_overall_score)}")
            print(f"  Saved incrementally: {save_path}")
        except Exception as error:
            # Record the failure, keep any previously saved partial results,
            # and continue to the next configuration. Never fabricate metrics.
            print(f"  CONFIG ERROR (recorded, continuing): {error}")
        print()

        # Delay between configurations to reduce rate-limit pressure.
        if config_index < len(configs) - 1 and args.delay > 0:
            time.sleep(args.delay)

    if not results:
        print("No experiment results were produced. Nothing to compare.")
        return

    comparison = compare_experiments(results)
    comparison_payload = comparison.to_dict()
    comparison_payload["pilot"] = args.max_questions is not None
    comparison_payload["questions_per_config"] = len(cases)
    comparison_payload["mode"] = args.mode
    comparison_payload["dataset"] = args.dataset
    comparison_path = RESULTS_DIR / comparison_filename
    comparison_path.write_text(
        json.dumps(comparison_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Saved comparison: {comparison_path}")
    print()

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 60)
    header_cols = (
        f"{'Configuration':<38} {'Chunk':>5} {'Ovl':>4} {'TopK':>4} "
        f"{'Answer':>7} {'Prec':>6} {'Recall':>6} {'F1':>6} "
        f"{'CtxRel':>7} {'Overall':>7} {'Lat(ms)':>8} {'APIerr':>6}"
    )
    print(header_cols)
    print("-" * len(header_cols))
    for result in results:
        avg_scores = result.diagnostics.get("average_scores", {})
        precision = avg_scores.get("retrieval_precision")
        recall = avg_scores.get("retrieval_recall")
        context_rel = avg_scores.get("context_relevance")
        row = (
            f"{result.config.name:<38} "
            f"{result.config.parameters.get('chunk_size', 'N/A'):>5} "
            f"{result.config.parameters.get('chunk_overlap', 'N/A'):>4} "
            f"{result.config.parameters.get('top_k', 'N/A'):>4} "
            f"{_fmt(result.average_answer_score):>7} "
            f"{_fmt(precision):>6} "
            f"{_fmt(recall):>6} "
            f"{_fmt(result.average_retrieval_score):>6} "
            f"{_fmt(context_rel):>7} "
            f"{_fmt(result.average_overall_score):>7} "
            f"{(f'{result.average_latency_ms:.0f}' if result.average_latency_ms is not None else 'N/A'):>8} "
            f"{count_api_errors(result):>6}"
        )
        print(row)
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
        score_str = _fmt(result.average_overall_score)
        print(f"  {i}. {result.config.name} — {score_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()