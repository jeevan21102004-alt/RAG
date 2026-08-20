from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from .agent import ANSWER, MAX_RETRIEVAL_ATTEMPTS, SEARCH, decide_initial_action, evaluate_context
from .chunking import chunk_documents
from .data_loader import load_documents
from .llm import generate_answer
from .retrieval import retrieve
from .vector_store import LocalVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "vector_store.json"


@dataclass(frozen=True)
class AgenticRunResult:
    question: str
    initial_decision: Any | None
    retrieval_attempts: int
    retrieved_results: list
    context_evaluations: list[Any]
    final_answer: str | None
    failure: str | None = None


def build_vector_store() -> LocalVectorStore:
    documents = load_documents()
    if not documents:
        raise RuntimeError("No documents found in the data/ directory.")

    chunks = chunk_documents(documents)
    store = LocalVectorStore.build_from_chunks(chunks)
    store.save(VECTOR_STORE_PATH)
    return store


def print_results(query: str, results) -> None:
    print(f"User question: {query}")
    print("Retrieved chunks:")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. score={result.score:.3f} | source={result.source} | chunk={result.chunk_index}")
        print(result.text)
        print()


def build_context(results) -> str:
    parts: list[str] = []
    for result in results:
        parts.append(f"[source={result.source} chunk={result.chunk_index} score={result.score:.3f}] {result.text}")
    return "\n\n".join(parts)


def print_agent_decision(decision) -> None:
    if decision is None:
        print("Agent decision: unavailable")
        print("Reason: The LLM did not return a valid decision.")
        print()
        return

    print(f"Agent decision: {decision.action}")
    print(f"Reason: {decision.reason}")
    print()


def print_final_answer(answer: str) -> None:
    print("Final answer:")
    print(answer)


def print_generation_error(error: Exception) -> None:
    print("Final answer:")
    print(f"LLM generation failed: {error}")


def _safe_generate_answer(question: str, context: str) -> str:
    return generate_answer(question, context)


def run_agentic_rag(question: str, top_k: int = 3, *, store: LocalVectorStore | None = None) -> AgenticRunResult:
    active_store = store or build_vector_store()
    try:
        initial_decision = decide_initial_action(question)
    except Exception as error:
        return AgenticRunResult(
            question=question,
            initial_decision=None,
            retrieval_attempts=0,
            retrieved_results=[],
            context_evaluations=[],
            final_answer=None,
            failure=f"Agent decision failed: {error}",
        )

    accumulated_results: list = []
    seen_chunks: set[tuple[str, int]] = set()
    evaluations: list[Any] = []
    context = ""

    if initial_decision.action == ANSWER:
        try:
            final_answer = _safe_generate_answer(question, context)
        except Exception as error:
            return AgenticRunResult(
                question=question,
                initial_decision=initial_decision,
                retrieval_attempts=0,
                retrieved_results=[],
                context_evaluations=[],
                final_answer=None,
                failure=f"LLM generation failed: {error}",
            )

        return AgenticRunResult(
            question=question,
            initial_decision=initial_decision,
            retrieval_attempts=0,
            retrieved_results=[],
            context_evaluations=[],
            final_answer=final_answer,
        )

    retrieval_attempts = 0
    while retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS:
        retrieval_attempts += 1
        attempt_top_k = top_k + retrieval_attempts - 1
        results = retrieve(question, active_store, top_k=attempt_top_k)

        for result in results:
            key = (result.source, result.chunk_index)
            if key not in seen_chunks:
                seen_chunks.add(key)
                accumulated_results.append(result)

        context = build_context(accumulated_results)
        try:
            evaluation = evaluate_context(question, context)
        except Exception as error:
            return AgenticRunResult(
                question=question,
                initial_decision=initial_decision,
                retrieval_attempts=retrieval_attempts,
                retrieved_results=accumulated_results,
                context_evaluations=evaluations,
                final_answer=None,
                failure=f"Context evaluation failed: {error}",
            )
        evaluations.append(evaluation)

        if evaluation.action == ANSWER:
            try:
                final_answer = _safe_generate_answer(question, context)
            except Exception as error:
                return AgenticRunResult(
                    question=question,
                    initial_decision=initial_decision,
                    retrieval_attempts=retrieval_attempts,
                    retrieved_results=accumulated_results,
                    context_evaluations=evaluations,
                    final_answer=None,
                    failure=f"LLM generation failed: {error}",
                )

            return AgenticRunResult(
                question=question,
                initial_decision=initial_decision,
                retrieval_attempts=retrieval_attempts,
                retrieved_results=accumulated_results,
                context_evaluations=evaluations,
                final_answer=final_answer,
            )

    try:
        final_answer = _safe_generate_answer(question, context)
    except Exception as error:
        return AgenticRunResult(
            question=question,
            initial_decision=initial_decision,
            retrieval_attempts=retrieval_attempts,
            retrieved_results=accumulated_results,
            context_evaluations=evaluations,
            final_answer=None,
            failure=f"LLM generation failed: {error}",
        )

    return AgenticRunResult(
        question=question,
        initial_decision=initial_decision,
        retrieval_attempts=retrieval_attempts,
        retrieved_results=accumulated_results,
        context_evaluations=evaluations,
        final_answer=final_answer,
    )


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Run the AdaptiveRAG baseline retrieval demo.")
    parser.add_argument("--query", required=True, help="Question to search for in the sample documents.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args(argv)

    store = build_vector_store()
    print(f"Question: {args.query}")
    print()

    result = run_agentic_rag(args.query, top_k=args.top_k, store=store)
    print_agent_decision(result.initial_decision)

    if result.failure and result.initial_decision is None:
        print_generation_error(Exception(result.failure))
        return

    if result.initial_decision.action == ANSWER:
        if result.failure:
            print_generation_error(Exception(result.failure))
            return
        if result.final_answer is not None:
            print_final_answer(result.final_answer)
        return

    for attempt_index, evaluation in enumerate(result.context_evaluations, start=1):
        print(f"Retrieval attempt: {attempt_index}")
        attempt_top_k = args.top_k + attempt_index - 1
        attempt_results = retrieve(args.query, store, top_k=attempt_top_k)
        print_results(args.query, attempt_results)
        print_agent_decision(evaluation)

    if result.retrieval_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        print("Maximum retrieval attempts reached; generating final answer with accumulated context.")
        print()

    if result.failure:
        print_generation_error(Exception(result.failure))
        return

    if result.final_answer is not None:
        print_final_answer(result.final_answer)
