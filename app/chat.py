"""Practical AdaptiveRAG chat interface (Phase 7B/7C).

Run with:  streamlit run app/chat.py

Thin UI over the existing ``RAGService`` (``src/adaptive_rag/rag_service.py``).
No second retrieval pipeline: all retrieval goes through ``query_rag`` with
the validated configuration (chunk_size=300, overlap=40, top_k=2).

Retrieval-only is the default. Optional LLM answer generation (Phase 7C)
is explicitly opt-in via a UI toggle and reuses the existing Gemini/LLM
integration through ``src/adaptive_rag/generation.py`` (which itself
delegates to ``llm.generate_answer``). The API key is read inside ``llm.py``
from the existing ``GEMINI_API_KEY`` configuration and is never exposed in
the UI or logs. No Gemini calls are made unless generation is enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_rag.generation import (  # noqa: E402
    GenerationResult,
    generate_grounded_answer,
)
from src.adaptive_rag.rag_service import (  # noqa: E402
    RAGResponse,
    get_validated_config,
    query_rag,
)

LLM_DISABLED_NOTICE = (
    "LLM answer generation is currently disabled -- "
    "this interface is retrieval-only (no Gemini/API calls)."
)

LLM_ENABLED_NOTICE = (
    "LLM answer generation is enabled -- answers are grounded ONLY in "
    "the retrieved context. Retrieval still comes from RAGService."
)


def handle_query(query: str) -> RAGResponse:
    """Handle one user query via RAGService (validated config, top_k=2)."""
    return query_rag(query)


def handle_query_with_generation(
    query: str,
    generate: bool = False,
    generate_fn: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Retrieve via RAGService and optionally generate a grounded answer.

    Retrieval ALWAYS comes from RAGService. When ``generate`` is False
    (default) no LLM call is made. When True, ONLY the retrieved context
    is passed to the generation layer.
    """
    response = query_rag(query)
    generation: GenerationResult | None = None
    if generate:
        generation = generate_grounded_answer(
            response.query, response.context, generate_fn=generate_fn)
    return {"response": response, "generation": generation}


def format_response(
    response: RAGResponse,
    generation: GenerationResult | None = None,
) -> dict[str, Any]:
    """Build display-ready sections from a RAGResponse (no Streamlit)."""
    chunk_lines = [
        f"[{i + 1}] {doc['source']}#chunk-{doc['chunk_index']} "
        f"(score={doc['score']:.3f})\n{doc['text']}"
        for i, doc in enumerate(response.retrieved_documents)
    ]
    metadata = response.retrieval_metadata
    metadata_line = (
        f"chunk_size={metadata.get('chunk_size')} | "
        f"overlap={metadata.get('chunk_overlap')} | "
        f"top_k={metadata.get('top_k_requested')} | "
        f"returned={metadata.get('num_returned')} | "
        f"latency={metadata.get('retrieval_latency_ms', 0.0):.1f} ms | "
        f"generation_used={bool(generation and generation.generation_used)}"
    )
    notice = (LLM_ENABLED_NOTICE if generation is not None
              else LLM_DISABLED_NOTICE)
    return {
        "query": response.query,
        "notice": notice,
        "sources": list(response.sources),
        "chunks": chunk_lines,
        "context": response.context,
        "answer": generation.answer if generation else None,
        "generation_used": bool(generation and generation.generation_used),
        "generation_error": generation.error if generation else None,
        "metadata": metadata_line,
        "config": get_validated_config(),
    }


def main() -> None:
    """Launch the Streamlit chat UI."""
    import streamlit as st

    st.set_page_config(page_title="AdaptiveRAG Chat", layout="wide")
    st.title("AdaptiveRAG Chat")
    st.info("Demo Mode -- No external API calls unless generation is enabled.")
    config = get_validated_config()
    st.caption(
        f"Validated config: chunk_size={config['chunk_size']}, "
        f"overlap={config['chunk_overlap']}, top_k={config['top_k']}"
    )
    enable_generation = st.toggle(
        "Enable LLM answer generation (uses GEMINI_API_KEY)",
        value=False,
        help="Off = retrieval-only. On = grounded answer from retrieved "
        "context via the existing Gemini integration.",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the enterprise knowledge base...")
    if not prompt:
        return
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        handled = handle_query_with_generation(
            prompt, generate=enable_generation)
        display = format_response(handled["response"], handled["generation"])
    except ValueError as error:
        with st.chat_message("assistant"):
            st.warning(str(error))
        st.session_state["messages"].append(
            {"role": "assistant", "content": str(error)})
        return

    with st.chat_message("assistant"):
        st.warning(display["notice"])
        if display["answer"]:
            st.subheader("Generated answer")
            st.markdown(display["answer"])
        elif display["generation_error"]:
            st.warning(
                "LLM generation failed (retrieval results below are "
                f"unaffected): {display['generation_error']}")
        st.subheader("Retrieved sources")
        st.write(", ".join(display["sources"]) or "No sources retrieved.")
        st.subheader("Retrieved chunks / context")
        for chunk in display["chunks"]:
            st.code(chunk[:2000])
        with st.expander("Retrieval metadata"):
            st.code(display["metadata"])

    st.session_state["messages"].append({
        "role": "assistant",
        "content": "Sources: " + (", ".join(display["sources"]) or "none")
        + "\n\n" + display["metadata"],
    })


if __name__ == "__main__":
    main()
