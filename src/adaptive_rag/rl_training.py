from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .evaluation import EvaluationQuestion, load_evaluation_dataset
from .policy import RetrievalPolicy
from .rl_environment import ACTION_ANSWER, ACTION_SEARCH, RetrievalDecisionEnvironment
from .state import question_to_state


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def create_train_test_split(
    dataset: list[EvaluationQuestion],
    test_ratio: float = 0.3,
    seed: int = SEED,
) -> tuple[list[EvaluationQuestion], list[EvaluationQuestion]]:
    """Split the dataset into training (70%) and test (30%) sets deterministically."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    test_size = int(len(dataset) * test_ratio)
    test_indices = set(indices[:test_size])
    train = [item for i, item in enumerate(dataset) if i not in test_indices]
    test = [item for i, item in enumerate(dataset) if i in test_indices]
    return train, test


@dataclass
class TrainingStats:
    episodes: int
    total_reward: float
    average_reward: float


def train_policy(
    train_questions: list[EvaluationQuestion],
    episodes: int = 2000,
    learning_rate: float = 0.01,
    seed: int = SEED,
) -> tuple[RetrievalPolicy, TrainingStats]:
    """Train a policy using the REINFORCE algorithm.

    loss = -log(probability_of_action) * reward
    """
    set_seed(seed)
    policy = RetrievalPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    total_reward = 0.0
    for episode in range(episodes):
        question = random.choice(train_questions)
        state = torch.tensor(question_to_state(question.question), dtype=torch.float32)

        probs = policy.action_probs(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        env = RetrievalDecisionEnvironment(expected_action=question.expected_action)
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

    stats = TrainingStats(
        episodes=episodes,
        total_reward=total_reward,
        average_reward=total_reward / episodes,
    )
    return policy, stats