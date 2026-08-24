"""Provider-independent system adapter interface.

This module defines a lightweight protocol that any AI system under
evaluation can implement.  The framework does NOT hardcode any specific
provider (e.g., Gemini, OpenAI).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .evaluation_schema import EvaluationCase, SystemResponse
from .experiment_config import ExperimentConfig


@runtime_checkable
class SystemAdapter(Protocol):
    """Protocol for an AI system that can be evaluated.

    Implementations should be provider-independent and deterministic where
    possible.  The adapter receives an :class:`EvaluationCase` and an
    :class:`ExperimentConfig` and returns a :class:`SystemResponse`.
    """

    def run(
        self,
        case: EvaluationCase,
        config: ExperimentConfig,
    ) -> SystemResponse:
        """Execute the AI system for a single evaluation case.

        Parameters
        ----------
        case:
            The evaluation case to process.
        config:
            The experiment configuration to use.

        Returns
        -------
        SystemResponse
            The system's response, including answer, action, retrieved
            documents, and metadata.
        """
        ...


class MockSystemAdapter:
    """A deterministic mock adapter for testing and demonstration.

    This adapter generates responses based on the configuration parameters
    without making any API calls.  It is fully deterministic and
    reproducible.

    Behavior:
    - If ``top_k`` >= 3, the mock performs a SEARCH action and retrieves
      documents.
    - If ``top_k`` < 3, the mock performs an ANSWER action directly.
    - The answer is derived from the expected answer (if available) or a
      generic response.
    - Latency is computed deterministically from the configuration.
    """

    def run(
        self,
        case: EvaluationCase,
        config: ExperimentConfig,
    ) -> SystemResponse:
        top_k = config.parameters.get("top_k", 3)
        chunk_size = config.parameters.get("chunk_size", 300)

        # Deterministic latency based on config
        latency_ms = float(top_k * 50 + chunk_size * 0.1)

        if top_k >= 3:
            action = "SEARCH"
            retrieved_documents = list(case.relevant_documents) if case.relevant_documents else []
            retrieved_context = case.expected_answer or "No context available."
            answer = case.expected_answer or "I don't know."
        else:
            action = "ANSWER"
            retrieved_documents = []
            retrieved_context = None
            answer = case.expected_answer or "I don't know."

        return SystemResponse(
            answer=answer,
            action=action,
            retrieved_documents=retrieved_documents,
            retrieved_context=retrieved_context,
            latency_ms=latency_ms,
            retrieval_attempts=1 if action == "SEARCH" else 0,
            metadata={
                "adapter": "mock",
                "config_top_k": top_k,
                "config_chunk_size": chunk_size,
            },
        )
