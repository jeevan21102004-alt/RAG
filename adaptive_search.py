"""CLI for Phase 3D adaptive retrieval configuration search.

Runs a deterministic heuristic adaptive search over the larger
``SearchSpace.adaptive_pilot()`` space (175 configurations), evaluating at
most ``--max-configurations`` of them.  Generation is disabled, so no
Gemini API calls are made.

Usage::

    python adaptive_search.py --dataset enterprise --pilot --max-questions 2

The pilot evaluates at most 8 configurations: a deterministic initial
exploration set (corners + centre) followed by neighbour-heuristic
selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.adaptive_search import (  # noqa: E402
    DEFAULT_EXPLORATION_BONUS,
    AdaptiveSearchEngine,
)
from src.adaptive_rag.objective import DEFAULT_WEIGHTS  # noqa: E402
from src.adaptive_rag.search_space import SearchSpace  # noqa: E402
from src.adaptive_rag.search_storage import save_search_result  # noqa: E402

SEARCH_RESULTS_DIR = PROJECT_ROOT / "experiments" / "search_results"


def _fmt(value):
    """Format a float for display, or N/A when missing."""
    return f"{value:.4f}" if value is not None else "N/A"


def build_search_space(pilot: bool) -> SearchSpace:
    """Return the deterministic search space for this run."""
    if pilot:
        return SearchSpace.adaptive_pilot()
    return SearchSpace.default()


def print_cli(space, weights, total, max_configurations, max_questions, bonus):
    """Print the search-space and objective summary block."""
    params = space.parameter_values
    print("========================================")
    print("ADAPTIVE RETRIEVAL SEARCH")
    print("========================================")
    print()
    print("Search space:")
    print(f"  Chunk sizes: {params.get('chunk_size', [])}")
    print(f"  Overlaps:    {params.get('chunk_overlap', [])}")
    print(f"  Top K:       {params.get('top_k', [])}")
    print(f"  Total possible configurations: {total}")
    print()
    print("Objective:")
    print(f"  F1 weight: {weights['retrieval_f1']:.2f}")
    print(f"  Context relevance weight: {weights['context_relevance']:.2f}")
    print(f"  Latency weight: {weights['latency']:.2f}")
    print()
    print("Adaptive strategy:")
    print(f"  Max configurations to evaluate: {max_configurations}")
    print(f"  Questions per configuration: {max_questions if max_questions is not None else 'all'}")
    print(f"  Exploration bonus: {bonus:.4f}")
    print()


def _entry_str(entry):
    return ", ".join(f"{k}={v}" for k, v in entry.parameters.items())


def print_evaluation_order(result):
    """Print each evaluated configuration with its selection reason."""
    print("========================================")
    print("EVALUATION ORDER")
    print("========================================")
    for index, entry in enumerate(result.evaluation_order, start=1):
        print(f"{index}. {entry.experiment_id}")
        print(f"   Parameters: {_entry_str(entry)}")
        print(f"   F1: {_fmt(entry.retrieval_f1)}")
        print(f"   Context relevance: {_fmt(entry.context_relevance)}")
        print(f"   Latency: {_fmt(entry.latency_ms)} ms")
        print(f"   Objective: {_fmt(entry.objective_score)}")
        print(f"   Status: {entry.status}")
        print(f"   Selection reason: {entry.selection_reason}")
        print()


def print_ranking(ranking):
    """Print the ranked search results."""
    print("========================================")
    print("SEARCH RESULTS (RANKED)")
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
    print("BEST CONFIGURATION (WITHIN EVALUATED SET)")
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
    """Entry point for the adaptive retrieval search CLI."""
    parser = argparse.ArgumentParser(
        description="Deterministic adaptive retrieval configuration search."
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
        help="Use the 175-configuration adaptive pilot space.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Deterministic cap on questions per configuration.",
    )
    parser.add_argument(
        "--max-configurations",
        type=int,
        default=8,
        help="Hard budget of configurations to evaluate (default 8).",
    )
    parser.add_argument(
        "--exploration-bonus",
        type=float,
        default=DEFAULT_EXPLORATION_BONUS,
        help="Priority bonus for exploration (default 0.05).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between questions. Default 0.",
    )
    args = parser.parse_args()

    space = build_search_space(args.pilot)
    total = space.count_combinations()

    print_cli(
        space,
        DEFAULT_WEIGHTS,
        total,
        args.max_configurations,
        args.max_questions,
        args.exploration_bonus,
    )

    engine = AdaptiveSearchEngine(
        space,
        dataset=args.dataset,
        max_questions=args.max_questions,
        max_configurations=args.max_configurations,
        exploration_bonus=args.exploration_bonus,
        delay=args.delay,
    )

    print("========================================")
    print("RUNNING ADAPTIVE SEARCH (SEQUENTIAL, API-FREE)")
    print("========================================")
    result = engine.run()

    print()
    print_evaluation_order(result)
    print_ranking(result.ranking)

    print()
    print(f"Configurations evaluated: {result.configurations_evaluated} / {result.total_possible_configurations}")
    print(f"Configurations remaining: {result.configurations_remaining}")
    print(f"Space coverage: {result.configurations_evaluated / result.total_possible_configurations * 100:.2f}%")
    print()

    save_path = save_search_result(result, SEARCH_RESULTS_DIR / f"{result.search_id}.json")
    print(f"Saved search result: {save_path}")
    print()

    print_best(result.best_configuration)

    return 0


if __name__ == "__main__":
    sys.exit(main())