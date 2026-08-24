# AdaptiveRAG Experiment Methodology

> **Status:** Phase 3B — Real AdaptiveRAG Experiments
> **Last updated:** 2026-08-24

This document describes the experimental methodology for comparing
different retrieval configurations of the real AdaptiveRAG system.

---

## ⚠️ Important Distinction

**Experimentation** ≠ **Optimization**

- **Experimentation:** Run known configurations and compare them.
- **Optimization:** Automatically search for a better configuration.

This phase implements **experimentation only**.  No automatic parameter
search is performed.

---

## Why These Parameters?

The three configurable parameters were chosen because they directly affect
retrieval quality:

- **chunk_size**: Controls how much text is in each retrieval unit.
  Larger chunks provide more context but may dilute relevance.
- **chunk_overlap**: Controls how much adjacent chunks overlap.
  More overlap reduces information loss at chunk boundaries.
- **top_k**: Controls how many chunks are retrieved per query.
  More chunks provide more context but increase latency and noise.

---

## Why These Four Configurations?

These configurations were selected manually as controlled experimental
conditions.  No automatic parameter search was performed.

| Config | chunk_size | chunk_overlap | top_k |
|--------|-----------|---------------|-------|
| A      | 100       | 20            | 3     |
| B      | 200       | 40            | 3     |
| C      | 300       | 50            | 5     |
| D      | 500       | 75            | 5     |

The configurations vary chunk_size (100–500) and top_k (3–5) to observe
how these parameters affect retrieval quality, answer quality, and latency.

---

## What Is Held Constant

- **Evaluation dataset**: `data/evaluation_questions.json` (20 questions)
- **Model**: The Gemini model configured in `.env`
- **Reward system**: Unchanged
- **Evaluation methodology**: The existing evaluation engine
- **Diagnostic thresholds**: Default thresholds from Phase 2
- **Random seed**: N/A (deterministic retrieval)

## What Changes Between Experiments

- `chunk_size`
- `chunk_overlap`
- `top_k`

---

## Evaluation Metrics

Each configuration is evaluated using the existing evaluation engine:

- **Answer Score**: Weighted combination of correctness, relevance,
  grounding, and completeness (0.0–1.0).
- **Retrieval Score**: F1 score of retrieved vs. relevant documents
  (0.0–1.0).
- **Decision Score**: Whether the system's action (SEARCH/ANSWER) matches
  the expected action (0.0 or 1.0).
- **Overall Score**: Weighted combination of answer, retrieval, and
  decision scores (0.0–1.0).
- **Latency**: Response time in milliseconds.

---

## Diagnostic Analysis

After evaluation, the diagnostic engine (Phase 2) is run on the results to
identify failure modes such as:

- Low retrieval recall/precision
- Low context relevance
- Low answer correctness/relevance/grounding/completeness
- Decision errors
- High latency
- Unnecessary retrieval

---

## API Quota Limitations

This experiment uses the real AdaptiveRAG pipeline, which makes Gemini API
calls.  If API quota is exhausted:

- Errors are handled gracefully.
- The experiment failure is recorded.
- No metrics are fabricated.
- No retries are attempted beyond the existing pipeline's retry logic.

---

## Result Storage

Results are saved as JSON files in `experiments/results/`:

- `adaptiverag_exp_a.json`
- `adaptiverag_exp_b.json`
- `adaptiverag_exp_c.json`
- `adaptiverag_exp_d.json`
- `adaptiverag_comparison.json`

No API keys or secrets are stored in result files.

---

## Limitations

- The expected answers are derived from document content and may not be
  exact matches.
- The mock adapter is not used for the real experiment — actual Gemini API
  calls are made.
- Results depend on API availability and quota.
- The four configurations are manually selected, not optimized.
