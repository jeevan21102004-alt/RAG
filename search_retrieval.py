"""CLI for Phase 3C automated retrieval configuration search.

Runs a deterministic grid search over a deterministic parameter space
defined in src.adaptive_rag.search_space, using the existing retrieval
benchmark and evaluation engine. Generation is disabled, so no Gemini
API calls are made.

Usage::

    python search_retrieval.py --dataset enterprise --pilot --max-questions 2

The --pilot flag uses a fixed 8-configuration search space;
--max-questions controls how many benchmark questions are used per
configuration.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.search_space import SearchSpace
from src.adaptive_rag.config_generator import generate_configurations
from src.adaptive_rag.objective import DEFAULT_WEIGHTS
from src.adaptive_rag.grid_search import GridSearchEngine
from src.adaptive_rag.search_storage import save_search_result

SEARCH_RESULTS_DIR = PROJECT_ROOT / "experiments" / "search_results"


def _fmt(value):
    """Format a float for display, or N/A when missing."""
    return f"{value:.4f}" if value is not None else "N/A"


def build_search_space(pilot):
    """Return the deterministic search space for this run."""
    if pilot:
        return SearchSpace.pilot()
    return SearchSpace.default()


def _entry_str(entry):
    """Render a ranking entry parameters as key=value, ..."""
    return ", ".join(f"{k}={v}" for k, v in entry.parameters.items())


def print_cli(space, weights, total_configurations, max_questions):
    """Print the search-space and objective summary block."""
    params = space.parameter_values
    print("========================================")
    print("AUTOMATED RETRIEVAL SEARCH")
    print("========================================")
    print()
    print("Search space:")
    print(f"  Chunk sizes: {params.get('chunk_size', [])}")
    print(f"  Overlaps:    {params.get('chunk_overlap', [])}")
    print(f"  Top K:       {params.get('top_k', [])}")
    print(f"  Total configurations: {total_configurations}")
    print()
    print("Objective:")
    print(f"  F1 weight: {weights['retrieval_f1']:.2f}")
    print(f"  Context relevance weight: {weights['context_relevance']:.2f}")
    print(f"  Latency weight: {weights['latency']:.2f}")
    print()
    print(f"  Questions per configuration: {max_questions if max_questions is not None else 'all'}")
    print()


def print_config_result(index, entry):
    """Print a single configuration evaluated summary."""
    print(f"Config {index}")
    print(f"  Parameters: {_entry_str(entry)}")
    print(f"  F1: {_fmt(entry.retrieval_f1)}")
    print(f"  Context relevance: {_fmt(entry.context_relevance)}")
    print(f"  Latency: {_fmt(entry.latency_ms)} ms")
    print(f"  Objective: {_fmt(entry.objective_score)}")
    print()


def print_ranking(ranking):
    """Print the ranked search results."""
    print("========================================")
    print("SEARCH RESULTS")
    print("========================================")
    for rank, entry in enumerate(ranking, start=1):
        print(f"Rank {rank}:")
        print(f"  Experiment: {entry.experiment_id}")
        print(f"  Parameters: {_entry_str(entry)}")
        print(f"  F1: {_fmt(entry.retrieval_f1)}")
        print(f"  Context relevance: {_fmt(entry.context_relevance)}")
        print(f"  Latency: {_fmt(entry.latency_ms)} ms")
        print(f"  Objective: {_fmt(entry.objective_score)}")
        print()


def print_best(best):
    """Print the best configuration block."""
    print("========================================")
    print("BEST CONFIGURATION")
    print("========================================")
    if best is None:
        print("No configuration completed successfully.")
        return
    print(f"  Experiment: {best.experiment_id}")
    print(f"  Parameters: {_entry_str(best)}")
    print(f"  F1: {_fmt(best.retrieval_f1)}")
    print(f"  Context relevance: {_fmt(best.context_relevance)}")
    print(f"  Latency: {_fmt(best.latency_ms)} ms")
    print(f"  Objective: {_fmt(best.objective_score)}")
    print("========================================")


def main():
    """Entry point for the automated retrieval search CLI."""
    parser = argparse.ArgumentParser(
        description="Deterministic automated retrieval configuration search."
    )
    parser.add_argument(
        "--dataset",
        choices=("enterprise", "default"),
        default="enterprise",
        help="Question set to search over (default: enterprise).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use the small deterministic 8-configuration search space.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Deterministic cap on questions per configuration.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between questions. Default 0 (API-free).",
    )
    args = parser.parse_args()

    space = build_search_space(args.pilot)
    configs = generate_configurations(space, prefix="grid")

    print_cli(
        space,
        DEFAULT_WEIGHTS,
        len(configs),
        args.max_questions,
    )

    engine = GridSearchEngine(
        space,
        dataset=args.dataset,
        max_questions=args.max_questions,
        delay=args.delay,
    )

    print("========================================")
    print("RUNNING GRID SEARCH (SEQUENTIAL, API-FREE)")
    print("========================================")
    result = engine.run()

    print()
    for index, entry in enumerate(result.ranking, start=1):
        print_config_result(index, entry)

    print_ranking(result.ranking)
    print()

    save_path = save_search_result(result, SEARCH_RESULTS_DIR / f"{result.search_id}.json")
    print(f"Saved search result: {save_path}")
    print()

    print_best(result.best_configuration)

    return 0


if __name__ == "__main__":
    sys.exit(main())
