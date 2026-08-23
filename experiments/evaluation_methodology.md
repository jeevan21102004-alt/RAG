# Evaluation Methodology

> **Status:** Phase 1B — Baseline Heuristic Evaluator
> **Last updated:** 2026-08-23

This document describes the evaluation methodology used by the AdaptiveRAG
AI Evaluation & Optimization Engine.  The current implementation provides
**baseline heuristic metrics** that are deterministic and require no
external API calls.

---

## ⚠️ Important Limitations

**Heuristic scores are not equivalent to human evaluation.**

The metrics below are computed using simple token-overlap and set-similarity
techniques.  They are designed to:

1. Establish a reproducible evaluation framework.
2. Provide a baseline that can be iterated on and improved.
3. Enable fast, deterministic evaluation without API costs.

They are **not** designed to:

- Replace human judgment of answer quality.
- Detect subtle hallucinations or factual inaccuracies.
- Capture semantic nuance, tone, or coherence.
- Replace LLM-as-judge or embedding-based similarity in future phases.

---

## Architecture

```
AI SYSTEM
    ↓
SystemResponse
    ↓
EVALUATION ENGINE (answer_evaluator + evaluation_runner)
    ↓
EvaluationResult
```

The evaluation engine is composed of two modules:

| Module | Responsibility |
|--------|---------------|
| `answer_evaluator.py` | Computes answer-quality dimensions and the weighted `answer_score`. |
| `evaluation_runner.py` | Orchestrates answer, retrieval, and decision scoring into a full `EvaluationResult`. |

---

## Answer-Quality Dimensions

All dimension scores use a consistent **0.0 – 1.0** range, where 0.0 is the
worst and 1.0 is the best.

### 1. Correctness Score

**What it measures:** How closely the actual answer matches the expected
answer in terms of content-word overlap.

**How it is calculated:**
- Tokenize both the expected answer and the actual answer (lowercase,
  remove punctuation, remove stop-words).
- Compute **Jaccard similarity** of the two token sets:
  `|expected ∩ actual| / |expected ∪ actual|`
- Score of 1.0 = identical token sets; 0.0 = no overlap.

**Scoring range:** 0.0 – 1.0

### 2. Relevance Score

**What it measures:** Whether the answer addresses the question by
containing important terms from the question.

**How it is calculated:**
- Tokenize the question and the actual answer.
- Compute **coverage ratio**: the fraction of question content-words that
  appear in the answer:
  `|question_terms ∩ answer_terms| / |question_terms|`
- If the question has no content-words, the score defaults to 1.0.

**Scoring range:** 0.0 – 1.0

### 3. Grounding Score

**What it measures:** Whether the answer is supported by the retrieved
context (i.e., the answer's terms appear in the retrieved documents).

**How it is calculated:**
- Tokenize the actual answer and the retrieved context.
- Compute **coverage ratio**: the fraction of answer content-words found in
  the retrieved context:
  `|answer_terms ∩ context_terms| / |answer_terms|`
- If no retrieved context is available, the score is **0.0** (grounding
  cannot be verified).

**Scoring range:** 0.0 – 1.0

### 4. Completeness Score

**What it measures:** Whether the actual answer covers the important terms
from the expected answer.

**How it is calculated:**
- Tokenize the expected answer and the actual answer.
- Compute **coverage ratio**: the fraction of expected-answer content-words
  present in the actual answer:
  `|expected_terms ∩ actual_terms| / |expected_terms|`
- If no expected answer is available, the score is **0.0**.

**Scoring range:** 0.0 – 1.0

---

## Answer-Score Weights

The `answer_score` is a weighted combination of the four dimensions:

| Dimension | Default Weight |
|-----------|---------------|
| Correctness | 0.40 |
| Relevance | 0.25 |
| Grounding | 0.25 |
| Completeness | 0.10 |
| **Total** | **1.00** |

```
answer_score = 0.40 × correctness + 0.25 × relevance + 0.25 × grounding + 0.10 × completeness
```

**Weights are configurable.**  Callers can pass a custom `weights` dict to
`evaluate_answer()` to override the defaults.

---

## Retrieval Score

**What it measures:** The fraction of relevant documents that were
successfully retrieved.

**How it is calculated:**
```
retrieval_score = |relevant ∩ retrieved| / |relevant|
```

**Edge-case behavior:**

| Scenario | Result |
|----------|--------|
| No relevant documents | `None` (cannot evaluate) |
| No retrieved documents | `0.0` (everything missed) |
| Perfect retrieval | `1.0` |
| Partial retrieval | Fraction of relevant docs retrieved |

**Scoring range:** 0.0 – 1.0 (or `None`)

---

## Decision Score

**What it measures:** Whether the system's action (SEARCH vs. ANSWER)
matches the expected action.

**How it is calculated:**
- If both `expected_action` and `actual_action` are present:
  - `1.0` if they match
  - `0.0` if they differ
- If either value is missing: `None` (cannot evaluate)

**Scoring range:** 0.0, 0.0, or `None`

---

## Overall Score

The `overall_score` combines the three component scores with proportional
weight redistribution.

**Default weights:**

| Component | Default Weight |
|-----------|---------------|
| Answer score | 0.60 |
| Retrieval score | 0.25 |
| Decision score | 0.15 |
| **Total** | **1.00** |

**Weight redistribution:**
If a component is `None` (unavailable), its weight is redistributed
**proportionally** among the remaining available components.  For example,
if retrieval_score is `None`:

```
overall = (answer_score × 0.60 + decision_score × 0.15) / (0.60 + 0.15)
```

If all components are `None`, the overall score is `None`.

**Scoring range:** 0.0 – 1.0 (or `None`)

---

## Component Score Storage

Individual component scores are stored in the `EvaluationResult.metadata`
field as a structured dictionary, so the existing `EvaluationResult` schema
is not modified.  The metadata contains:

- `answer_scores`: Full breakdown of correctness, relevance, grounding,
  completeness, and the weighted answer_score.
- `retrieval_score`: The retrieval score (or `None`).
- `decision_score`: The decision score (or `None`).
- `overall_weights`: The weights used for the overall score calculation.

---

## Future Improvements

This baseline evaluator is intentionally simple.  Future phases may add:

- **LLM-as-judge** for nuanced answer quality assessment.
- **Embedding-based similarity** for more robust correctness scoring.
- **Hallucination detection** using cross-encoder models.
- **Semantic grounding** checks using claim extraction.
- **Configurable metric pipelines** for different evaluation scenarios.

These improvements will build on the framework established here without
breaking the existing data contract.
