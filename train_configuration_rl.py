"""Training entry point for RL configuration selection (Phase 4).

Runs REINFORCE over the enterprise retrieval benchmark using a
generation-disabled adapter.  **No Gemini calls are made** — the entire
pipeline (training split, retrieval, evaluation, reward) is local.

Usage::

    python train_configuration_rl.py --episodes 5 --budget 4 --seed 42

Saves:
    models/configuration_rl_policy.pt
    models/configuration_rl_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.adaptive_rag.evaluation_schema import EvaluationCase  # noqa: E402
from src.adaptive_rag.grid_search import ENTERPRISE_KB_DIR, load_search_cases  # noqa: E402
from src.adaptive_rag.optimization_training import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    ReinforceTrainer,
)
from src.adaptive_rag.retrieval_benchmark_adapter import (  # noqa: E402
    RetrievalBenchmarkAdapter,
)
from src.adaptive_rag.search_space import SearchSpace  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
POLICY_PATH = MODELS_DIR / "configuration_rl_policy.pt"
METADATA_PATH = MODELS_DIR / "configuration_rl_metadata.json"


def load_split(path: Path) -> tuple[list[str], list[str]]:
    """Load the deterministic train/test case-ID split."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        list(data["training_case_ids"]),
        list(data["test_case_ids"]),
    )


def load_cases_by_ids() -> dict[str, EvaluationCase]:
    """Load every enterprise question and index it by case_id."""
    all_cases = load_search_cases("enterprise", max_questions=None)
    return {case.case_id: case for case in all_cases}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="REINFORCE training for RL configuration selection."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "enterprise_rl_split.json")
    args = parser.parse_args()

    space = SearchSpace.adaptive_pilot()
    train_ids, test_ids = load_split(args.split)
    cases_by_id = load_cases_by_ids()
    training_cases = [cases_by_id[cid] for cid in train_ids]
    test_cases = [cases_by_id[cid] for cid in test_ids]

    adapter = RetrievalBenchmarkAdapter(
        enable_generation=False, data_dir=ENTERPRISE_KB_DIR
    )

    print("============================================")
    print("RL CONFIGURATION SELECTION — TRAINING (SMOKE)")
    print("============================================")
    print(f"Search space: {space.count_combinations()} configurations")
    print(f"Training questions: {len(training_cases)}")
    print(f"Test questions: {len(test_cases)}")
    print(f"Episodes: {args.episodes}   Budget: {args.budget}   Seed: {args.seed}")
    print(f"Learning rate: {args.learning_rate}   Gamma: {args.gamma}")
    print("Generation disabled: True (zero Gemini calls)")
    print()

    trainer = ReinforceTrainer(
        search_space=space,
        cases=training_cases,
        adapter=adapter,
        episodes=args.episodes,
        budget=args.budget,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )
    summary = trainer.train()

    print("Training complete.")
    print(f"  Losses finite: {summary['loss_finite']}")
    for i, loss in enumerate(summary["losses"], start=1):
        print(f"  Episode {i}: loss={loss:.6f}  reward={summary['episode_rewards'][i-1]:.4f}"
              f"  best_obj={summary['episode_objectives'][i-1]}")
    print(f"  Best objective across episodes: {summary['best_episode_objective']}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch_save_state = {
        "policy_state_dict": trainer.policy.state_dict(),
        "n_actions": trainer.n_actions,
        "state_dim": trainer.policy.state_dim,
        "hidden_dim": trainer.hidden_dim,
    }
    import torch

    torch.save(torch_save_state, POLICY_PATH)
    metadata = {
        "architecture": {
            "type": "MLP",
            "state_dim": trainer.policy.state_dim,
            "hidden_dim": trainer.hidden_dim,
            "n_actions": trainer.n_actions,
            "layers": [
                f"Linear({trainer.policy.state_dim},{trainer.hidden_dim})",
                "ReLU",
                f"Linear({trainer.hidden_dim},{trainer.n_actions})",
            ],
            "activation": "ReLU",
            "output": "Softmax over action IDs",
        },
        "seed": args.seed,
        "episodes": args.episodes,
        "budget": args.budget,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "search_space": space.to_dict(),
        "training_split": train_ids,
        "test_split": test_ids,
        "loss_finite": summary["loss_finite"],
        "final_loss": summary["final_loss"],
        "api_calls": 0,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved policy: {POLICY_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print("ZERO Gemini/API calls made during training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())