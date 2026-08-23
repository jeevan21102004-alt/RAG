from pathlib import Path

import torch

from src.adaptive_rag.agent import ANSWER, SEARCH
from src.adaptive_rag.balanced_reward import calculate_balanced_reward
from src.adaptive_rag.evaluation import load_evaluation_dataset
from src.adaptive_rag.policy import RetrievalPolicy
from src.adaptive_rag.rl_training import SEED, create_train_test_split
from src.adaptive_rag.semantic_rl_environment import ACTION_ANSWER, ACTION_SEARCH, SemanticRetrievalDecisionEnvironment
from src.adaptive_rag.semantic_state import embedding_dimension, question_to_embedding


MODEL_PATH = Path("models") / "semantic_retrieval_policy.pt"


def evaluate_policy(policy: RetrievalPolicy, questions) -> dict:
    correct = 0
    search_correct = 0
    search_total = 0
    answer_correct = 0
    answer_total = 0
    total_reward = 0.0
    search_count = 0
    answer_count = 0

    for question in questions:
        state = torch.tensor(question_to_embedding(question.question), dtype=torch.float32)
        probs = policy.action_probs(state)
        action = int(torch.argmax(probs).item())

        env = SemanticRetrievalDecisionEnvironment(
            expected_action=question.expected_action,
            reward_function=calculate_balanced_reward,
        )
        env.reset(question.question)
        _, reward, _, _ = env.step(action)

        total_reward += reward
        if action == ACTION_SEARCH:
            search_count += 1
        else:
            answer_count += 1

        if question.expected_action == SEARCH:
            search_total += 1
            if action == ACTION_SEARCH:
                search_correct += 1
                correct += 1
        elif question.expected_action == ANSWER:
            answer_total += 1
            if action == ACTION_ANSWER:
                answer_correct += 1
                correct += 1

    n = len(questions)
    return {
        "n": n,
        "accuracy": correct / n * 100.0 if n else 0.0,
        "search_accuracy": search_correct / search_total * 100.0 if search_total else 0.0,
        "answer_accuracy": answer_correct / answer_total * 100.0 if answer_total else 0.0,
        "average_reward": total_reward / n if n else 0.0,
        "search_rate": search_count / n * 100.0 if n else 0.0,
        "answer_rate": answer_count / n * 100.0 if n else 0.0,
    }


def majority_baseline_accuracy(questions) -> float:
    search_count = sum(1 for q in questions if q.expected_action == SEARCH)
    answer_count = sum(1 for q in questions if q.expected_action == ANSWER)
    majority = max(search_count, answer_count)
    return majority / len(questions) * 100.0 if questions else 0.0


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run `python train_semantic_rl.py` first.")
        return

    dataset = load_evaluation_dataset()
    _, test_questions = create_train_test_split(dataset, seed=SEED)

    dim = embedding_dimension()
    policy = RetrievalPolicy(input_dim=dim)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    policy.eval()

    results = evaluate_policy(policy, test_questions)
    baseline = majority_baseline_accuracy(test_questions)

    collapsed_to_search = results["search_rate"] == 100.0
    collapsed_to_answer = results["answer_rate"] == 100.0

    print("========================================")
    print("Semantic RL Policy Evaluation (Test Set)")
    print("============================")
    print(f"Test questions: {results['n']}")
    print()
    print(f"RL Policy Accuracy: {results['accuracy']:.2f}%")
    print(f"SEARCH Accuracy: {results['search_accuracy']:.2f}%")
    print(f"ANSWER Accuracy: {results['answer_accuracy']:.2f}%")
    print(f"Average Reward: {results['average_reward']:.3f}")
    print(f"SEARCH Rate: {results['search_rate']:.2f}%")
    print(f"ANSWER Rate: {results['answer_rate']:.2f}%")
    print()
    print(f"Majority Baseline Accuracy: {baseline:.2f}%")
    print()
    if collapsed_to_search:
        print("WARNING: Policy collapsed to always SEARCH.")
    if collapsed_to_answer:
        print("WARNING: Policy collapsed to always ANSWER.")
    print()
    print("--- Experiment Comparison ---")
    print("Experiment 1 (5-feature, asymmetric reward):")
    print("  Test Accuracy: 66.67% | Majority Baseline: 66.67% | SEARCH Rate: 100%")
    print("Experiment 2 (semantic, balanced reward):")
    print(f"  Test Accuracy: {results['accuracy']:.2f}% | Majority Baseline: {baseline:.2f}% | SEARCH Rate: {results['search_rate']:.2f}%")
    print()
    if results["accuracy"] > baseline:
        print("Experiment 2 BEATS the majority baseline.")
    else:
        print("Experiment 2 does NOT beat the majority baseline.")
    print("========================================")


if __name__ == "__main__":
    main()