"""Practical AdaptiveRAG chat interface (Phase 7B) -- retrieval-only.

Run with:  streamlit run app/chat.py

Thin UI over the existing ``RAGService`` (``src/adaptive_rag/rag_service.py``).
No second retrieval pipeline: all retrieval goes through ``query_rag`` with
the validated configuration (chunk_size=300, overlap=40, top_k=2).

Retrieval-only: LLM answer generation is disabled. No Gemini calls, no
external API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_rag.rag_service import (  # noqa: E402
    RAGResponse,
    get_validated_config,
    query_rag,
)

LLM_DISABLED_NOTICE = (
    "LLM answer generation is currently disabled -- "
    "this interface is retrieval-only (no Gemini/API calls)."
)


def handle_query(query: str) -> RAGResponse:
    """Handle one user query via RAGService (validated config, top_k=2)."""
    return query_rag(query)


def format_response(response: RAGResponse) -> dict[str, Any]:
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
        f"latency={metadata.get('retrieval_latency_ms', 0.0):.1f} ms"
    )
    return {
        "query": response.query,
        "notice": LLM_DISABLED_NOTICE,
        "sources": list(response.sources),
        "chunks": chunk_lines,
        "context": response.context,
        "metadata": metadata_line,
        "config": get_validated_config(),
    }


def main() -> None:
    """Launch the Streamlit chat UI."""
    import streamlit as st

    st.set_page_config(page_title="AdaptiveRAG Chat", layout="wide")
    st.title("AdaptiveRAG Chat (Retrieval-Only)")
    st.info("Demo Mode -- No external API calls. " + LLM_DISABLED_NOTICE)
    config = get_validated_config()
    st.caption(
        f"Validated config: chunk_size={config['chunk_size']}, "
        f"overlap={config['chunk_overlap']}, top_k={config['top_k']}"
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
        display = format_response(handle_query(prompt))
    except ValueError as error:
        with st.chat_message("assistant"):
            st.warning(str(error))
        st.session_state["messages"].append(
            {"role": "assistant", "content": str(error)})
        return

    with st.chat_message("assistant"):
        st.warning(display["notice"])
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
