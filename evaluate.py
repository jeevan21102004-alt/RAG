from src.adaptive_rag.evaluation import format_record, run_evaluation, save_results


def main() -> None:
    records, metrics = run_evaluation()

    print("# ========================================")
    print("AdaptiveRAG Controlled Evaluation")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Successful Runs: {metrics['successful_runs']}")
    print(f"API Errors: {metrics['api_errors']}")
    print()

    for index, record in enumerate(records, start=1):
        print(format_record(index, record))

    print("# ========================================")
    print("RESULTS")
    print(f"Overall Decision Accuracy: {metrics['overall_decision_accuracy']:.2f}%")
    print()
    print(f"Retrieval Required Accuracy: {metrics['retrieval_required_accuracy']:.2f}%")
    print(f"Retrieval Not Required Accuracy: {metrics['retrieval_not_required_accuracy']:.2f}%")
    print(f"Insufficient Context Accuracy: {metrics['insufficient_context_accuracy']:.2f}%")
    print()
    print(f"Unnecessary Retrieval Rate: {metrics['unnecessary_retrieval_rate']:.2f}%")
    print(f"Missed Retrieval Rate: {metrics['missed_retrieval_rate']:.2f}%")
    print(f"Average Retrieval Attempts: {metrics['average_retrieval_attempts']:.2f}")
    print(f"Retrieval Success Rate: {metrics['retrieval_success_rate']:.2f}%")

    save_results(records, metrics)


if __name__ == "__main__":
    main()
