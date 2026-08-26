"""Evaluation entry point for RL configuration selection (Phase 4).

Compares Random, Grid, Adaptive, and RL configuration selection on the
held-out test questions using the **same** search space, budget, dataset,
and objective.

Usage::

    python evaluate_configuration_rl.py --budget 4 --seed 42

Generation is disabled throughout — zero Gemini/API calls.  The RL method
uses the trained policy from ``models/configuration_rl_policy.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.configuration_policy import ConfigurationPolicy  # noqa: E402
from src.adaptive_rag.optimization_baselines import (  # noqa: E402
    AdaptiveSearchBaseline,
    GridSearchBaseline,
    RandomSearchBaseline,
)
from src.adaptive_rag.optimization_state import (  # noqa: E402
    optimization_state_to_vector,
)
from src.adaptive_rag.rl_optimization_environment import (  # noqa: E402
    RetrievalOptimizationEnvironment,
)
from src.adaptive_rag.optimization_result import (  # noqa: E402
    OptimizationEntry,
    OptimizationResult,
)
from src.adaptive_rag.grid_search import (  # noqa: E402
    ENTERPRISE_KB_DIR,
    load_search_cases,
)
from src.adaptive_rag.retrieval_benchmark_adapter import (  # noqa: E402
    RetrievalBenchmarkAdapter,
)
from src.adaptive_rag.search_space import SearchSpace  # noqa: E402

POLICY_PATH = PROJECT_ROOT / "models" / "configuration_rl_policy.pt"


def evaluate_rl_policy(
    space: SearchSpace,
    cases,
    adapter,
    budget: int,
    policy,
    device,
    seed: int,
    test_ids,
) -> OptimizationResult:
    """Run the trained policy greedily (deterministic) on *cases*."""
    env = RetrievalOptimizationEnvironment(
        search_space=space,
        cases=cases,
        adapter=adapter,
        budget=budget,
    )
    state = env.reset()
    order: list[OptimizationEntry] = []
    done = False
    while not done:
        state_tensor = torch.tensor(
            optimization_state_to_vector(state),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        valid_mask = torch.tensor(
            env.valid_action_mask(), dtype=torch.bool, device=device
        ).unsqueeze(0)
        action, _ = policy.act(state_tensor, valid_mask, deterministic=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    best_entry: OptimizationEntry | None = None
    best_obj = best_f1 = best_ctx = best_lat = None
    for step, record in enumerate(env._order, start=1):
        entry = OptimizationEntry(
            method="rl",
            step=step,
            action_id=record["action_id"],
            parameters=dict(record["parameters"]),
            objective_score=record["objective"],
            retrieval_f1=record["f1"],
            context_relevance=record["context_relevance"],
            latency_ms=record["latency_ms"],
            reward=record["reward"],
            status=record["status"],
        )
        order.append(entry)
        if record["objective"] is not None and (
            best_obj is None or record["objective"] > best_obj
        ):
            best_obj, best_entry = record["objective"], entry
            best_f1 = record["f1"]
            best_ctx = record["context_relevance"]
            best_lat = record["latency_ms"]

    return OptimizationResult(
        method="rl",
        seed=seed,
        budget=budget,
        search_space=space.to_dict(),
        test_question_ids=list(test_ids),
        configurations_evaluated=len(order),
        evaluation_order=order,
        best_configuration=best_entry,
        best_objective=best_obj,
        best_f1=best_f1,
        best_context_relevance=best_ctx,
        best_latency_ms=best_lat,
        cumulative_reward=env.cumulative_reward,
        metadata={"policy": str(POLICY_PATH)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare configuration-selection methods on test questions."
    )
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "enterprise_rl_split.json",
    )
    args = parser.parse_args()

    space = SearchSpace.adaptive_pilot()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_ids = split["test_case_ids"]
    all_cases = {
        c.case_id: c for c in load_search_cases("enterprise", max_questions=None)
    }
    test_cases = [all_cases[cid] for cid in test_ids]

    adapter = RetrievalBenchmarkAdapter(
        enable_generation=False, data_dir=ENTERPRISE_KB_DIR
    )

    if not POLICY_PATH.exists():
        print("No trained policy found. Train first:")
        print("  python train_configuration_rl.py --episodes 5 --budget 4 --seed 42")
        return 1

    checkpoint = torch.load(POLICY_PATH, map_location="cpu", weights_only=False)
    policy = ConfigurationPolicy(
        n_actions=checkpoint["n_actions"],
        state_dim=checkpoint["state_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        seed=args.seed,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    device = torch.device("cpu")

    results = []

    print("Running Random baseline...")
    results.append(
        RandomSearchBaseline(space, args.budget, args.seed, adapter, test_cases).run()
    )
    print("Running Grid baseline...")
    results.append(GridSearchBaseline(space, args.budget, adapter, test_cases).run())
    print("Running Adaptive baseline...")
    results.append(
        AdaptiveSearchBaseline(space, args.budget, adapter, test_cases).run()
    )
    print("Running RL policy...")
    results.append(
        evaluate_rl_policy(
            space, test_cases, adapter, args.budget, policy, device, args.seed, test_ids
        )
    )

    print()
    print("========================================")
    print("CONFIGURATION OPTIMIZATION BENCHMARK")
    print("========================================")
    header = (
        f"{'Method':<10} {'Budget':<7} {'Best Obj':<10} {'Best F1':<9} "
        f"{'CtxRel':<8} {'Latency':<9} {'Eval':<5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.method:<10} {r.budget:<7} "
            f"{(f'{r.best_objective:.4f}' if r.best_objective is not None else 'N/A'):<10} "
            f"{(f'{r.best_f1:.4f}' if r.best_f1 is not None else 'N/A'):<9} "
            f"{(f'{r.best_context_relevance:.4f}' if r.best_context_relevance is not None else 'N/A'):<8} "
            f"{(f'{r.best_latency_ms:.2f}' if r.best_latency_ms is not None else 'N/A'):<9} "
            f"{r.configurations_evaluated:<5}"
        )

    print()
    valid = [r for r in results if r.best_objective is not None]
    if valid:
        best = max(valid, key=lambda r: r.best_objective)
        print(f"BEST METHOD (by best objective on test questions): {best.method}")
        print(f"  best objective {best.best_objective:.4f}")
        print()
        print("NOTE: This is a budget-4 smoke test. Do not conclude any")
        print("method is superior from a handful of configurations.")
    else:
        print("No method produced a measurable objective.")

    return 0


if __name__ == "__main__":
    sys.exit(main())