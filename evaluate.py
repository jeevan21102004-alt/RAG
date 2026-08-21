import argparse

from src.adaptive_rag.evaluation import format_record, run_evaluation, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdaptiveRAG controlled evaluation.")
    parser.add_argument("--quick", action="store_true", help="Run the quick 6-question evaluation.")
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    records, metrics = run_evaluation(quick=args.quick)

    print("========================================")
    print("AdaptiveRAG Quick Evaluation" if args.quick else "AdaptiveRAG Controlled Evaluation")
    print("============================")
    print()
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Successful Runs: {metrics['successful_runs']}")
    print(f"API Errors: {metrics['api_errors']}")
    print()
    print(f"Overall Decision Accuracy: {metrics['overall_decision_accuracy']:.2f}%")
    print()
    print(f"Retrieval Required Accuracy: {metrics['retrieval_required_accuracy']:.2f}%")
    print(f"Retrieval Not Required Accuracy: {metrics['retrieval_not_required_accuracy']:.2f}%")
    print(f"Insufficient Context Accuracy: {metrics['insufficient_context_accuracy']:.2f}%")
    print()
    print(f"Unnecessary Retrieval Rate: {metrics['unnecessary_retrieval_rate']:.2f}%")
    print(f"Missed Retrieval Rate: {metrics['missed_retrieval_rate']:.2f}%")
    print(f"Average Retrieval Attempts: {metrics['average_retrieval_attempts']:.2f}")
    print()
    print("========================================")

    save_results(records, metrics, mode=mode)


if __name__ == "__main__":
    main()