# Controlled Retrieval Benchmark Methodology

> **Status:** Phase 3B — Controlled Retrieval Benchmark (Forced SEARCH)
> **Last updated:** 2026-08-24

---

## Why Forced Retrieval Is Necessary

The Phase 3B agentic pilot revealed a **confounding issue**: the Agentic
RAG decision layer selected ANSWER for the first two questions in every
configuration, so the retrieval parameters (chunk_size, chunk_overlap,
top_k) were never exercised.  The pilot therefore could not be used to
conclude which retrieval configuration is best.

The controlled retrieval benchmark removes this confound by **forcing
SEARCH** for every question, ensuring retrieval is always executed and its
quality is directly measurable.

**"The controlled retrieval benchmark evaluates retrieval independently of
the agent's SEARCH/ANSWER decision."**

---

## Why the Agentic Decision Layer Is Excluded

The production agent must remain unchanged:

- `agent.py`, `policy.py`, `reward.py`, `balanced_reward.py`,
  `rl_environment.py`, `semantic_rl_environment.py`, `semantic_state.py`
  are NOT modified.
- `python main.py --query "..."` behaves exactly as before.
- The agent still independently chooses SEARCH or ANSWER in production.

The benchmark uses a separate adapter
(`retrieval_benchmark_adapter.py`) that bypasses only the decision step.
This isolates the variable under study: the retrieval configuration.

```
Question → FORCED SEARCH → Retriever → Configuration → Evaluation
```

---

## Dataset Selection

`data/retrieval_benchmark_questions.json` contains **8 questions**, all of
which require information from the provided documents.

- Source: the existing `data/evaluation_questions.json`.
- Selection rule: all questions with `category == "retrieval_required"`
  and `expected_action == "SEARCH"` (8 available; target was 10).
- No invented questions — every question is supported by document content.
- Expected answers are derived verbatim from the source documents.

Each case retains: `case_id`, `question`, `category`, `expected_answer`,
`expected_action`, `relevant_documents`.

---

## Configuration Matrix

The same four fixed configurations are reused (no automatic search):

| Config | chunk_size | chunk_overlap | top_k |
|--------|-----------|---------------|-------|
| A      | 100       | 20            | 3     |
| B      | 200       | 40            | 3     |
| C      | 300       | 50            | 5     |
| D      | 500       | 75            | 5     |

Held constant: dataset, model, evaluation methodology, diagnostic
thresholds, delay between questions.  Only the three retrieval parameters
change.

---

## Primary Metric

**Retrieval F1 is the primary metric** for this benchmark.

Reported metrics per configuration:

- Retrieval precision / recall / F1 (primary)
- Context relevance (lexical coverage heuristic)
- Answer score (only when generation is enabled)
- Average latency (split into retrieval vs. generation)
- API error count

Agent decision accuracy is intentionally NOT used as the primary metric
because SEARCH is forced by design.

---

## Latency Measurement

Latency components are measured separately and stored in metadata:

- `retrieval_latency_ms`: time spent inside the retriever.
- `generation_latency_ms`: time spent generating the answer with Gemini
  (`None` when generation is disabled).
- `total_latency_ms`: sum of the two.

Retrieval latency must never be mixed with LLM generation latency when
comparing configurations.

---

## Limitations

- The lexical context-relevance score is a baseline heuristic, not a
  semantic measure.
- The corpus is tiny (3 documents), so differences between configurations
  may be small; results should be treated as directional, not definitive.
- Generation depends on Gemini availability; API errors are recorded and
  never retried excessively or hidden.
- These configurations were selected manually as controlled experimental
  conditions.  No automatic parameter search was performed.

---

## Experimentation ≠ Optimization

- **Experimentation:** run known configurations and compare them.
- **Optimization:** automatically search for better configurations.

This benchmark performs experimentation only.  Optimization comes later.