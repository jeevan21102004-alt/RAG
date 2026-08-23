"""Local demonstration of the RL retrieval decision environment.

This script does NOT call Gemini or any external API.
It loads questions from the evaluation dataset, converts them to states,
takes example actions, and prints the resulting rewards.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_rag.rl_environment import (
    ACTION_ANSWER,
    ACTION_SEARCH,
    ACTION_TO_LABEL,
    RetrievalDecisionEnvironment,
)
from src.adaptive_rag.state import state_feature_names


def load_questions() -> list[dict]:
    dataset_path = PROJECT_ROOT / "data" / "evaluation_questions.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def main() -> None:
    questions = load_questions()
    feature_names = state_feature_names()

    print("========================================")
    print("RL Retrieval Decision Environment Demo")
    print("============================")
    print()

    # Take a few example questions across categories.
    example_indices = [0, 8, 14]
    example_actions = [ACTION_SEARCH, ACTION_ANSWER, ACTION_SEARCH]

    for index, action in zip(example_indices, example_actions):
        item = questions[index]
        question = item["question"]
        expected_action = item["expected_action"]

        env = RetrievalDecisionEnvironment(expected_action=expected_action)
        state = env.reset(question)

        print(f"Question: {question}")
        print(f"Expected action: {expected_action}")
        print("State features:")
        for name, value in zip(feature_names, state):
            print(f"  {name}: {value}")

        next_state, reward, done, metadata = env.step(action)
        print(f"Action taken: {ACTION_TO_LABEL[action]} ({action})")
        print(f"Reward: {reward:+.1f}")
        print(f"Done: {done}")
        print(f"Reason: {metadata['reward_reason']}")
        print()

    print("========================================")


if __name__ == "__main__":
    main()