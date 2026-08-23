"""Local demonstration of the deterministic reward system.

This script does NOT call Gemini or any external API.
It simply demonstrates the reward calculation for the four action pairs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_rag.agent import ANSWER, SEARCH
from src.adaptive_rag.reward import calculate_reward


def main() -> None:
    cases = [
        (SEARCH, SEARCH),
        (SEARCH, ANSWER),
        (ANSWER, SEARCH),
        (ANSWER, ANSWER),
    ]

    print("========================================")
    print("Deterministic Reward Demonstration")
    print("============================")
    print()

    for expected_action, actual_action in cases:
        result = calculate_reward(expected_action, actual_action)
        print(f"Expected: {result.expected_action}")
        print(f"Actual:   {result.actual_action}")
        print(f"Reward:   {result.reward:+d}")
        print(f"Reason:   {result.reason}")
        print()

    print("========================================")


if __name__ == "__main__":
    main()