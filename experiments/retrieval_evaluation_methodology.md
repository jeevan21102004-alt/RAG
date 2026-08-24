# Retrieval Evaluation Methodology

> **Status:** Phase 1C — Baseline Heuristic Retrieval Evaluator
> **Last updated:** 2026-08-24

This document describes the retrieval-quality evaluation methodology used
by the AdaptiveRAG AI Evaluation & Optimization Engine.  The current
implementation provides **baseline heuristic metrics** that are
deterministic and require no external API calls.

---

## ⚠️ Important Limitations

**Lexical context relevance is a baseline heuristic and does not represent
semantic relevance.**

The metrics in this module are computed using simple token-overlap and
set-similarity techniques.  They are designed to:

1. Establish a reproducible evaluation framework.
2. Provide a baseline that can be iterated on and improved.
3. Enable fast, deterministic evaluation without API costs.

They are **not** designed to:

- Replace human judgment of retrieval quality.
- Capture semantic relevance or topical coherence.
- Replace embedding-based or LLM-based retrieval evaluation in future phases.

---

## Architecture

```
AI SYSTEM
    ↓
SystemResponse (retrieved_documents, retrieved_context)
    ↓
RETRIEVAL EVALUATOR (retrieval_evaluator.py)
    ↓
RetrievalMetrics (precision, recall, F1)
    ↓
EVALUATION RUNNER (evaluation_runner.py)
    ↓
EvaluationResult (retrieval_score = F1, metadata with full metrics)
```

---

## Precision

**What it measures:** Of all documents retrieved, what fraction were
actually relevant?

**Formula:**
```
precision = |relevant ∩ retrieved| / |retrieved|
```

**Edge cases:**
- No retrieved documents → precision = 0.0
- All retrieved documents are relevant → precision = 1.0

**Scoring range:** 0.0 – 1.0

---

## Recall

**What it measures:** Of all relevant documents, what fraction were
retrieved?

**Formula:**
```
recall = |relevant ∩ retrieved| / |relevant|
```

**Edge cases:**
- No relevant documents → recall = 0.0
- All relevant documents retrieved → recall = 1.0

**Scoring range:** 0.0 – 1.0

---

## F1 Score

**What it measures:** The harmonic mean of precision and recall, balancing
both concerns.

**Formula:**
```
F1 = 2 × precision × recall / (precision + recall)
```

**Edge cases:**
- Both precision and recall are 0 → F1 = 0.0
- Perfect precision and recall → F1 = 1.0

**Scoring range:** 0.0 – 1.0

**The F1 score is used as the primary `retrieval_score`** in
`EvaluationResult`.

---

## Top-K Evaluation

The evaluator supports calculating metrics at different cutoffs (top-1,
top-3, top-5) via `calculate_retrieval_metrics_at_k()`.

**How it works:**
- Only the first *k* retrieved documents are considered.
- If *k* exceeds the number of retrieved documents, all retrieved
  documents are used.
- Precision, recall, and F1 are then calculated over this subset.

**Use case:** Evaluating retrieval quality at different list sizes to
understand how ranking quality degrades as more documents are considered.

---

## Context Relevance

**What it measures:** What fraction of the question's content-words appear
in the retrieved context text?

**Formula:**
```
context_relevance = |question_terms ∩ context_terms| / |question_terms|
```

**How it is calculated:**
- Tokenize the question (lowercase, remove punctuation, remove stop-words).
- Tokenize the retrieved context (same normalization).
- Compute the coverage ratio: fraction of question content-words found in
  the context.

**Edge cases:**
- No retrieved context → score = 0.0
- Question has no content-words → score = 1.0

**Scoring range:** 0.0 – 1.0

.. warning::
    This is a **baseline heuristic lexical metric** and does NOT represent
    semantic relevance.  It only checks for term overlap.

---

## Document ID Normalization

Document identifiers may differ in formatting (e.g., path prefixes, file
extensions, casing).  The `normalize_doc_id()` function standardizes
identifiers before comparison.

**Normalization rules:**
1. Convert to lower-case.
2. Strip leading/trailing whitespace.
3. Remove common path prefixes: `data/`, `docs/`, `./`, `../`.
4. Remove the `.md` extension if present.
5. Extract the core alphanumeric token sequence and join with underscores.

**Examples:**

| Input | Normalized |
|-------|-----------|
| `data/supervised_learning.md` | `supervised_learning` |
| `supervised_learning.md` | `supervised_learning` |
| `./docs/Machine_Learning.md` | `machine_learning` |
| `overfitting` | `overfitting` |

**Important:** This is **not** fuzzy matching.  Two IDs are considered equal
only when their normalized forms are identical.

---

## Duplicate Handling

Duplicate document IDs in the retrieved list are removed before
calculation.  This prevents systems from inflating precision by returning
the same document multiple times.

---

## Integration with EvaluationResult

The retrieval metrics are integrated into the evaluation pipeline as
follows:

- `retrieval_score` field: Set to the **F1 score** (primary retrieval
  metric).
- `metadata["retrieval_metrics"]`: Full `RetrievalMetrics` dict (precision,
  recall, F1, counts).
- `metadata["context_relevance"]`: The context relevance score.
- `metadata["retrieval_score"]`: The F1-based retrieval score.

The existing `EvaluationResult` schema is **not modified** — all new
information is stored in the `metadata` field.

---

## Overall Score Integration

The `retrieval_score` (F1) participates in the overall score calculation
with a default weight of 0.25:

```
overall = 0.60 × answer_score + 0.25 × retrieval_score + 0.15 × decision_score
```

If the retrieval score is unavailable (`None`), its weight is redistributed
proportionally among the remaining available components.

---

## Future Improvements

This baseline evaluator is intentionally simple.  Future phases may add:

- **Embedding-based retrieval evaluation** (e.g., MRR, nDCG).
- **Semantic context relevance** using cross-encoders.
- **LLM-as-judge** for retrieval quality assessment.
- **Multi-vector retrieval** evaluation.

These improvements will build on the framework established here without
breaking the existing data contract.
