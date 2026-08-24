"""Generic experiment runner.

This module orchestrates the execution of an AI system under a given
configuration across multiple evaluation cases, evaluates the responses,
runs diagnostics, and produces an aggregated :class:`ExperimentResult`.

The runner is **provider-independent** and makes **no API calls** itself.
It delegates execution to a :class:`SystemAdapter`.
"""

from __future__ import annotations

import time
from typing import Any

from .diagnostics import diagnose_system
from .evaluation_runner import run_evaluation
from .evaluation_schema import EvaluationCase, EvaluationResult
from .experiment_config import ExperimentConfig
from .experiment_result import ExperimentResult
from .system_adapter import SystemAdapter


def run_experiment(
    config: ExperimentConfig,
    cases: list[EvaluationCase],
    adapter: SystemAdapter,
) -> ExperimentResult:
    """Run an experiment and produce an aggregated :class:`ExperimentResult`.

    Steps
    -----
    1. Execute every evaluation case through the adapter.
    2. Measure execution latency if not already provided by the adapter.
    3. Convert each response into an :class:`EvaluationResult` using the
       existing evaluation runner.
    4. Aggregate metrics (averages, counts).
    5. Run the diagnostic engine on the collected results.
    6. Produce and return an :class:`ExperimentResult`.

    Parameters
    ----------
    config:
        The experiment configuration.
    cases:
        List of evaluation cases to run.
    adapter:
        A :class:`SystemAdapter` that executes the AI system.

    Returns
    -------
    ExperimentResult
        Aggregated experiment results with diagnostics.
    """
    evaluation_results: list[EvaluationResult] = []

    for case in cases:
        # Execute the AI system
        start_time = time.perf_counter()
        response = adapter.run(case, config)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Use measured latency if the adapter didn't provide one
        if response.latency_ms is None:
            response = response.__class__(
                answer=response.answer,
                action=response.action,
                retrieved_documents=response.retrieved_documents,
                retrieved_context=response.retrieved_context,
                latency_ms=elapsed_ms,
                retrieval_attempts=response.retrieval_attempts,
                metadata=response.metadata,
            )

        # Evaluate the response
        result = run_evaluation(case, response)
        evaluation_results.append(result)

    # Aggregate metrics
    total_cases = len(evaluation_results)
    successful_cases = sum(1 for r in evaluation_results if r.status == "SUCCESS")
    failed_cases = total_cases - successful_cases

    def _avg(values: list[float | None]) -> float | None:
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    answer_scores = [r.answer_score for r in evaluation_results]
    retrieval_scores = [r.retrieval_score for r in evaluation_results]
    decision_scores = [r.decision_score for r in evaluation_results]
    overall_scores = [r.overall_score for r in evaluation_results]
    latencies = [r.latency_ms for r in evaluation_results]
    retrieval_attempts = [r.retrieval_attempts for r in evaluation_results]

    # Run diagnostics
    system_diagnosis = diagnose_system(evaluation_results)

    return ExperimentResult(
        experiment_id=config.experiment_id,
        config=config,
        total_cases=total_cases,
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        average_answer_score=_avg(answer_scores),
        average_retrieval_score=_avg(retrieval_scores),
        average_decision_score=_avg(decision_scores),
        average_overall_score=_avg(overall_scores),
        average_latency_ms=_avg(latencies),
        total_retrieval_attempts=sum(retrieval_attempts),
        diagnostics=system_diagnosis.to_dict(),
        metadata={
            "evaluation_results": [r.to_dict() for r in evaluation_results],
        },
    )
