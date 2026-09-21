"""AdaptiveRAG optimization dashboard (Phase 5) -- DEMO MODE by default.

Run with:  streamlit run app/dashboard.py

Zero external API calls in demo mode. All metrics come from the local
enterprise corpus + local retrieval evaluation, or from the labelled
Phase 4 smoke-test record. No evaluation/diagnostic logic lives here;
everything delegates to ``src.adaptive_rag.dashboard_service``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_rag.dashboard_service import (  # noqa: E402
    compare_search_methods,
    demo_questions,
    documents_by_category,
    get_recommended_configuration,
    list_configurations,
    load_benchmark_questions,
    load_enterprise_documents,
    load_rl_split,
    run_diagnosis,
    run_retrieval_evaluation,
    search_space_summary,
)

st.set_page_config(page_title="AdaptiveRAG", layout="wide")
st.sidebar.title("AdaptiveRAG")
st.sidebar.caption("Demo Mode -- No external API calls")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Knowledge Base", "Retrieval Evaluation",
     "Failure Diagnosis", "Experiments", "Optimizer",
     "Configuration Explorer"],
)


def _demo_banner() -> None:
    st.info("🟢 Demo Mode — No external API calls. "
            "All metrics are local or from saved records.")


if page == "Dashboard":
    _demo_banner()
    st.title("AdaptiveRAG")
    st.subheader("Evaluate, diagnose and optimize enterprise AI "
                 "retrieval systems.")
    st.write("AdaptiveRAG is an evaluation and optimization framework "
             "for enterprise Retrieval-Augmented Generation systems.")
    docs = load_enterprise_documents()
    questions = load_benchmark_questions()
    space = search_space_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Knowledge Base", f"{len(docs)} Documents")
    c2.metric("Benchmark", f"{len(questions)} Questions")
    c3.metric("Search Space", f"{space['total_configurations']} Configs")
    c4.metric("Evaluation Tests", "321 in prior baseline")
    st.write("Knowledge Base → Retrieval → Evaluation → "
             "Diagnosis → Optimization")
    st.bar_chart(documents_by_category())


elif page == "Knowledge Base":
    _demo_banner()
    st.header("Knowledge Base")
    docs = load_enterprise_documents()
    st.write(f"Total documents: {len(docs)}")
    st.bar_chart(documents_by_category())
    cats = sorted({d["category"] for d in docs})
    cat = st.selectbox("Category", cats)
    names = [d["name"] for d in docs if d["category"] == cat]
    name = st.selectbox("Document", names)
    doc = next(d for d in docs if d["name"] == name)
    st.write(f"Words: {doc['words']} | Path: {doc['path']}")
    st.code(doc["text"][:6000])


elif page == "Retrieval Evaluation":
    _demo_banner()
    st.header("Retrieval Evaluation")
    st.caption("Retrieval-only. Generation disabled -- no API calls.")
    questions = load_benchmark_questions()
    qid = st.selectbox("Question",
                       [q["case_id"] for q in questions],
                       format_func=lambda c: c + " -- " + next(
                           q["question"] for q in questions
                           if q["case_id"] == c)[:70])
    chunk_size = st.slider("Chunk Size", 50, 500, 300, step=50)
    chunk_overlap = st.slider("Chunk Overlap", 10, 75, 50, step=5)
    top_k = st.slider("Top K", 2, 6, 3)
    if st.button("Evaluate Retrieval"):
        with st.spinner("Running local retrieval evaluation..."):
            ev = run_retrieval_evaluation(qid, chunk_size,
                                          chunk_overlap, top_k)
        st.session_state["last_eval"] = ev
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{ev['precision']:.4f}"
                  if ev["precision"] is not None else "n/a")
        m2.metric("Recall", f"{ev['recall']:.4f}"
                  if ev["recall"] is not None else "n/a")
        m3.metric("F1", f"{ev['f1']:.4f}"
                  if ev["f1"] is not None else "n/a")
        m4.metric("Latency (ms)",
                  f"{ev['retrieval_latency_ms']:.1f}"
                  if ev["retrieval_latency_ms"] else "n/a")
        st.write("Context relevance:",
                 ev["context_relevance"])
        st.write("Expected:", ev["expected_documents"])
        st.write("Retrieved:", ev["retrieved_documents"])


elif page == "Failure Diagnosis":
    _demo_banner()
    st.header("Failure Diagnosis")
    ev = st.session_state.get("last_eval")
    if ev is None:
        st.write("Run a Retrieval Evaluation first, or use the demo below.")
        questions = demo_questions()
        qid = st.selectbox("Demo question",
                           [q["case_id"] for q in questions])
        if st.button("Diagnose demo evaluation"):
            with st.spinner("Evaluating + diagnosing locally..."):
                ev = run_retrieval_evaluation(qid, 300, 50, 5)
                st.session_state["last_eval"] = ev
    if ev is not None:
        diag = run_diagnosis(ev)
        st.write("Overall status:", diag["overall_status"])
        st.write("Precision:", ev["precision"], "| Recall:",
                 ev["recall"], "| F1:", ev["f1"])
        for f in diag["findings"]:
            with st.expander(f["failure_type"] + " [" + f["severity"] + "]"):
                st.write("Score:", f["score"], "| Threshold:",
                         f["threshold"])
                st.write("Evidence:", f["evidence"])
                st.write("Cause:", f["message"])
                st.write("Recommendation:", f["recommendation"])
        if not diag["findings"]:
            st.success("No issues detected by the diagnostic engine.")


elif page == "Experiments":
    _demo_banner()
    st.header("Experiments")
    comp = compare_search_methods()
    st.caption(comp["label"] + " -- saved record, not a live run.")
    st.table(comp["rows"])
    st.success("Best measured method: " + comp["best_method"] +
               " (do NOT claim RL is better).")
    st.bar_chart({r["method"]: r["best_objective"] for r in comp["rows"]})


elif page == "Optimizer":
    _demo_banner()
    st.header("Optimizer")
    rec = get_recommended_configuration()
    st.caption("Source: " + rec["source"])
    st.write("CURRENT:", rec["current"])
    st.write("RECOMMENDED:", rec["recommended"])
    st.info(rec["reason"])


elif page == "Configuration Explorer":
    _demo_banner()
    st.header("Configuration Explorer")
    rows = list_configurations()
    st.write(f"{len(rows)} configurations (175 search space). "
             "Unevaluated rows show 'Not evaluated' -- no fake metrics.")
    st.dataframe([{"ID": r["action_id"], "chunk_size": r["chunk_size"],
                   "overlap": r["chunk_overlap"], "top_k": r["top_k"],
                   "F1": "Not evaluated", "Objective": "Not evaluated"}
                  for r in rows[:50]])
    st.caption("Showing first 50 of 175. Sort/filter in full product.")



