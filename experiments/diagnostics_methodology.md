# Diagnostics Methodology

> **Status:** Phase 2 — Deterministic AI Failure Diagnosis Engine
> **Last updated:** 2026-08-24

This document describes the failure diagnosis methodology used by the
AdaptiveRAG AI Evaluation & Optimization Engine.  The current implementation
provides **deterministic diagnostic heuristics** that require no external
API calls.

---

## ⚠️ Important Limitations

**Diagnostic heuristics are deterministic rules and do NOT prove causal
relationships.**

The diagnostics in this module are designed to:

1. Surface likely failure modes for human investigation.
2. Provide actionable recommendations for debugging.
3. Enable systematic analysis of evaluation results.

They are **not** designed to:

- Replace root-cause analysis by a human expert.
- Guarantee causal conclusions.
- Replace LLM-based or statistical diagnostic methods.

---

## Architecture

```
EvaluationResult
       ↓
diagnose_result()
       ↓
DiagnosticFinding[]

Multiple EvaluationResults
       ↓
diagnose_system()
       ↓
SystemDiagnosis
```

The diagnostic layer is **separate** from the evaluation pipeline.  It
consumes `EvaluationResult` objects and produces diagnostic findings.

---

## Failure Types

| Failure Type | Description |
|-------------|-------------|
| `RETRIEVAL_LOW_RECALL` | Retriever missed relevant documents. |
| `RETRIEVAL_LOW_PRECISION` | Retriever returned too many irrelevant documents. |
| `CONTEXT_LOW_RELEVANCE` | Retrieved context does not address the question. |
| `ANSWER_LOW_CORRECTNESS` | Answer does not match the expected answer. |
| `ANSWER_LOW_RELEVANCE` | Answer does not address the question. |
| `ANSWER_LOW_GROUNDING` | Answer contains claims not supported by context. |
| `ANSWER_LOW_COMPLETENESS` | Answer is missing key information. |
| `DECISION_ERROR` | System action does not match expected action. |
| `HIGH_LATENCY` | Response time exceeds acceptable threshold. |
| `UNNECESSARY_RETRIEVAL` | System retrieved when a direct answer was expected. |
| `UNKNOWN` | Unrecognized failure mode. |

---

## Thresholds

All thresholds are configurable via the `thresholds` parameter.  Default
values:

| Metric | Default Threshold | Direction |
|--------|------------------|-----------|
| Retrieval recall | 0.70 | Below = failure |
| Retrieval precision | 0.70 | Below = failure |
| Context relevance | 0.60 | Below = failure |
| Answer correctness | 0.70 | Below = failure |
| Answer relevance | 0.70 | Below = failure |
| Answer grounding | 0.70 | Below = failure |
| Answer completeness | 0.70 | Below = failure |
| Decision score | 1.0 | Below = failure |
| Latency (ms) | 2000.0 | Above = failure |

---

## Severity Calculation

Severity is determined by how far a metric deviates from its threshold.

### For metrics below threshold (e.g., recall, precision, correctness):

| Gap (threshold - score) | Severity |
|------------------------|----------|
| >= 0.30 | CRITICAL |
| >= 0.15 | HIGH |
| >= 0.05 | MEDIUM |
| < 0.05 | LOW |

### For metrics above threshold (e.g., latency):

| Excess (score - threshold) | Severity |
|---------------------------|----------|
| >= 1000 ms | CRITICAL |
| >= 500 ms | HIGH |
| >= 250 ms | MEDIUM |
| < 250 ms | LOW |

### Special cases:
- **Decision errors** are always `HIGH` severity.
- **Unnecessary retrieval** is always `MEDIUM` severity.

---

## Root-Cause Heuristics

The system-level diagnosis applies deterministic heuristics to suggest
likely root causes.  These are **not guaranteed causal conclusions**.

### Heuristic 1: Retrieval Bottleneck
**Condition:** Low recall (< 0.70) AND low answer correctness (< 0.70)
**Suggestion:** "Retrieval may be the primary bottleneck: low recall
combined with low answer correctness suggests the retriever is missing
relevant documents."

### Heuristic 2: Generation Bottleneck
**Condition:** High recall (>= 0.70) AND low correctness (< 0.70) AND low
grounding (< 0.70)
**Suggestion:** "Generation may not be using retrieved evidence
effectively: high recall but low correctness and low grounding suggest the
generator is not leveraging retrieved context."

### Heuristic 3: Excessive Irrelevant Context
**Condition:** Low precision (< 0.70) AND low context relevance (< 0.60)
**Suggestion:** "Retriever may be returning excessive irrelevant context:
low precision combined with low context relevance."

### Heuristic 4: Decision Policy Issues
**Condition:** Any decision errors detected
**Suggestion:** "Retrieval policy may require optimization: frequent
decision errors detected."

---

## System-Level Aggregation

`diagnose_system()` aggregates findings across multiple evaluation results:

- **total_cases**: Total number of results analyzed.
- **successful_cases**: Results with status SUCCESS.
- **failed_cases**: Results with non-SUCCESS status.
- **failure_counts**: Mapping of FailureType → count.
- **average_scores**: Average of each available metric across all results.
- **top_failure_modes**: Failure types ranked by frequency (count and
  percentage).
- **recommendations**: Aggregated root-cause heuristic suggestions.

Failure modes are ranked by frequency (descending).

---

## Diagnostic Finding Structure

Each `DiagnosticFinding` contains:

- `failure_type`: The type of failure detected.
- `severity`: LOW, MEDIUM, HIGH, or CRITICAL.
- `score`: The actual metric value (may be `None` for action-based findings).
- `threshold`: The threshold that was violated.
- `message`: Human-readable description of what happened.
- `evidence`: Supporting evidence for the finding.
- `recommendation`: Suggested next steps for investigation.

---

## Future Improvements

This baseline diagnostic engine is intentionally simple.  Future phases may
add:

- **Statistical anomaly detection** for identifying unusual patterns.
- **Correlation analysis** between failure modes.
- **LLM-based diagnostic reasoning** for nuanced root-cause analysis.
- **Interactive diagnostic workflows** with drill-down capabilities.

These improvements will build on the framework established here without
breaking the existing data contract.
