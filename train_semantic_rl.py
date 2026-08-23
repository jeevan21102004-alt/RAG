import json
import random
from pathlib import Path

import torch

from src.adaptive_rag.balanced_reward import calculate_balanced_reward
from src.adaptive_rag.evaluation import load_evaluation_dataset
from src.adaptive_rag.policy import RetrievalPolicy
from src.adaptive_rag.rl_training import SEED, create_train_test_split, set_seed
from src.adaptive_rag.semantic_rl_environment import SemanticRetrievalDecisionEnvironment
from src.adaptive_rag.semantic_state import EMBEDDING_MODEL_NAME, embedding_dimension, question_to_embedding


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "semantic_retrieval_policy.pt"
EXPERIMENT_PATH = MODEL_DIR / "semantic_experiment.json"


def train_semantic_policy(
    train_questions,
    input_dim: int,
    episodes: int = 2000,
    learning_rate: float = 0.01,
    seed: int = SEED,
) -> tuple[RetrievalPolicy, float]:
    """Train a policy using semantic embeddings and the balanced reward."""
    set_seed(seed)
    policy = RetrievalPolicy(input_dim=input_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    total_reward = 0.0
    for episode in range(episodes):
        question = random.choice(train_questions)
        state = torch.tensor(question_to_embedding(question.question), dtype=torch.float32)

        probs = policy.action_probs(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        env = SemanticRetrievalDecisionEnvironment(
            expected_action=question.expected_action,
            reward_function=calculate_balanced_reward,
        )
        env.reset(question.question)
        _, reward, _, _ = env.step(int(action.item()))

        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward

        if (episode + 1) % 500 == 0:
            avg = total_reward / (episode + 1)
            print(f"Episode {episode + 1}/{episodes} | Average reward: {avg:.3f}")

    return policy, total_reward


def main() -> None:
    set_seed(SEED)
    dataset = load_evaluation_dataset()
    train_questions, test_questions = create_train_test_split(dataset, seed=SEED)

    dim = embedding_dimension()
    print("========================================")
    print("Semantic RL Policy Training (REINFORCE)")
    print("============================")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension: {dim}")
    print(f"Reward scheme: balanced (+1 correct, -1 unnecessary SEARCH, -1 missed SEARCH)")
    print(f"Seed: {SEED}")
    print(f"Training questions: {len(train_questions)}")
    print(f"Test questions: {len(test_questions)}")
    print()

    policy, total_reward = train_semantic_policy(train_questions, input_dim=dim, seed=SEED)

    print()
    print(f"Training complete: 2000 episodes")
    print(f"Total training reward: {total_reward:.2f}")
    print(f"Average training reward: {total_reward / 2000:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"Policy saved to: {MODEL_PATH}")

    experiment = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": dim,
        "reward_scheme": "balanced (+1 correct, -1 unnecessary SEARCH, -1 missed SEARCH)",
        "random_seed": SEED,
        "train_question_ids": [q.question for q in train_questions],
        "test_question_ids": [q.question for q in test_questions],
        "episodes": 2000,
        "optimizer": "Adam",
        "learning_rate": 0.01,
    }
    EXPERIMENT_PATH.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print(f"Experiment metadata saved to: {EXPERIMENT_PATH}")


if __name__ == "__main__":
    main()