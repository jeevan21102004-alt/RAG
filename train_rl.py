from pathlib import Path

import torch

from src.adaptive_rag.evaluation import load_evaluation_dataset
from src.adaptive_rag.rl_training import SEED, create_train_test_split, set_seed, train_policy


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "retrieval_policy.pt"


def main() -> None:
    set_seed(SEED)
    dataset = load_evaluation_dataset()
    train_questions, test_questions = create_train_test_split(dataset, seed=SEED)

    print("========================================")
    print("AdaptiveRAG RL Policy Training (REINFORCE)")
    print("============================")
    print(f"Seed: {SEED}")
    print(f"Total questions: {len(dataset)}")
    print(f"Training questions: {len(train_questions)}")
    print(f"Test questions: {len(test_questions)}")
    print()

    policy, stats = train_policy(train_questions, episodes=2000, seed=SEED)

    print()
    print(f"Training complete: {stats.episodes} episodes")
    print(f"Total training reward: {stats.total_reward:.2f}")
    print(f"Average training reward: {stats.average_reward:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"Policy saved to: {MODEL_PATH}")

    # Save split information for reproducibility.
    split_path = Path("models") / "split_info.json"
    import json

    split_info = {
        "seed": SEED,
        "train_questions": [q.question for q in train_questions],
        "test_questions": [q.question for q in test_questions],
    }
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"Split info saved to: {split_path}")


if __name__ == "__main__":
    main()