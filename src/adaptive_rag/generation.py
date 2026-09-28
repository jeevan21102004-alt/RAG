"""Optional LLM answer generation (Phase 7C).

Small generation layer over context already retrieved by RAGService.
Retrieval-only remains the default; generation is explicitly opt-in.

Reuses the existing Gemini/LLM integration in ``llm.py`` (``generate_answer``)
-- no second LLM client, no new dependency. The API key is read inside
``llm.py`` from the existing ``GEMINI_API_KEY`` configuration and is never
exposed here, in logs, or in the UI.

Strict grounding: the model is instructed to answer ONLY from the supplied
context and to say explicitly when the context is insufficient. Empty
context never triggers an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The retrieved context does not contain enough information "
    "to answer this question."
)


@dataclass(frozen=True)
class GenerationResult:
    """Result of an (optional) grounded generation call."""

    query: str
    answer: str | None = None
    generation_used: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_grounding_prompt(query: str, context: str) -> str:
    """Construct a strict grounding prompt for *query* over *context*."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    cleaned_query = query.strip()
    cleaned_context = context.strip()
    return (
        "You are a careful assistant for a retrieval-augmented generation "
        "system.\n"
        "Answer the user's question using ONLY the retrieved context below.\n"
        "Do NOT use outside knowledge. Do NOT invent facts.\n"
        "If the retrieved context does not contain enough information to "
        "answer, say so explicitly: "
        f"'{INSUFFICIENT_CONTEXT_MESSAGE}'\n"
        "Keep the answer concise and grounded in the context.\n\n"
        f"Retrieved context:\n{cleaned_context}\n\n"
        f"Question: {cleaned_query}\n"
        "Grounded answer:"
    )


def generate_grounded_answer(
    query: str,
    context: str,
    generate_fn: Callable[[str, str], str] | None = None,
) -> GenerationResult:
    """Generate a grounded answer for *query* from *context*.

    ``generate_fn`` defaults to the existing ``llm.generate_answer`` and is
    imported lazily so this module never requires an API key at import time.
    Pass a stub in tests -- no Gemini calls are made unless the caller opts
    in with a real function and a configured key.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    cleaned_query = query.strip()
    if not isinstance(context, str) or not context.strip():
        return GenerationResult(
            query=cleaned_query,
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            generation_used=False,
            error="empty context: generation skipped",
            metadata={"reason": "empty-context"},
        )
    if generate_fn is None:
        from .llm import generate_answer as generate_fn  # lazy, opt-in only

    prompt = build_grounding_prompt(cleaned_query, context.strip())
    try:
        answer = generate_fn(cleaned_query, context.strip())
    except Exception as error:  # graceful failure, never raise to the UI
        return GenerationResult(
            query=cleaned_query,
            answer=None,
            generation_used=False,
            error=str(error),
            metadata={"reason": "generation-failed"},
        )
    if not isinstance(answer, str) or not answer.strip():
        return GenerationResult(
            query=cleaned_query,
            answer=None,
            generation_used=False,
            error="generator returned an empty answer",
            metadata={"reason": "empty-answer"},
        )
    return GenerationResult(
        query=cleaned_query,
        answer=answer.strip(),
        generation_used=True,
        error=None,
        metadata={"prompt_chars": len(prompt)},
    )
