# Experiment Framework Methodology

> **Status:** Phase 3A — Experiment Framework
> **Last updated:** 2026-08-24

This document describes the experiment framework for the AdaptiveRAG AI
Evaluation & Optimization Engine.

---

## ⚠️ Important Distinction

**Experimentation** ≠ **Optimization**

- **Experimentation:** Run known configurations and compare them.
- **Optimization:** Automatically search for a better configuration.

This phase implements **experimentation only**.  Optimization comes later.

---

## Architecture

```
CONFIGURATION
      ↓
AI SYSTEM (via SystemAdapter)
      ↓
SYSTEM RESPONSES
      ↓
EVALUATION ENGINE (evaluation_runner)
      ↓
DIAGNOSTIC ENGINE (diagnostics)
      ↓
EXPERIMENT RESULT
      ↓
EXPERIMENT COMPARISON
```

---

## ExperimentConfig

An :class:`ExperimentConfig` describes a set of parameters that define how
an AI system should be run during an experiment.

Fields:
- `experiment_id`: Unique identifier.
- `name`: Human-readable name.
- `description`: Optional description.
- `parameters`: Dictionary of arbitrary configuration values (e.g.,
  `top_k`, `chunk_size`, `retrieval_strategy`).
- `tags`: Optional list of tags for categorization.

The configuration is:
- JSON serializable
- Immutable (frozen dataclass)
- Provider-independent
- Independent of Gemini

---

## ExperimentResult

An :class:`ExperimentResult` aggregates the outcomes of running an
experiment across multiple evaluation cases.

Fields:
- `experiment_id`: Identifier matching the config.
- `config`: The configuration used.
- `total_cases`: Total number of cases run.
- `successful_cases`: Cases that completed without error.
- `failed_cases`: Cases with non-SUCCESS status.
- `average_answer_score`: Mean answer score.
- `average_retrieval_score`: Mean retrieval score (F1).
- `average_decision_score`: Mean decision score.
- `average_overall_score`: Mean overall score.
- `average_latency_ms`: Mean latency in milliseconds.
- `total_retrieval_attempts`: Sum of retrieval attempts.
- `diagnostics`: Aggregated system diagnosis.
- `metadata`: Additional metadata (including per-case evaluation results).

---

## Experiment Runner

The :func:`run_experiment` function:

1. Executes every evaluation case through a :class:`SystemAdapter`.
2. Measures execution latency if not already provided.
3. Converts each response into an :class:`EvaluationResult` using the
   existing evaluation runner.
4. Aggregates metrics (averages, counts).
5. Runs the diagnostic engine on the collected results.
6. Produces and returns an :class:`ExperimentResult`.

The runner is **provider-independent** and makes **no API calls** itself.

---

## System Adapter

A :class:`SystemAdapter` is a lightweight protocol that any AI system can
implement:

```python
def run(case: EvaluationCase, config: ExperimentConfig) -> SystemResponse
```

The framework does NOT hardcode any specific provider.  A
:class:`MockSystemAdapter` is provided for testing and demonstration.

---

## Storage

Experiment results are stored as JSON files in `experiments/results/`.

Functions:
- `save_experiment_result()`: Save to JSON.
- `load_experiment_result()`: Load from JSON.

No database is used.  No secrets are stored.

---

## Experiment Comparison

The :func:`compare_experiments` function compares multiple
:class:`ExperimentResult` objects:

- `best_by_overall_score`: Highest average overall score.
- `best_by_answer_score`: Highest average answer score.
- `best_by_retrieval_score`: Highest average retrieval score.
- `fastest_experiment`: Lowest average latency.
- `ranking`: Experiment IDs ranked by average overall score (descending).

Experiments with `None` scores are ranked last.

---

## What This Phase Does NOT Do

- Does NOT perform automatic parameter search.
- Does NOT optimize configurations.
- Does NOT use Bayesian optimization.
- Does NOT use RL optimization.
- Does NOT use LangChain, LangGraph, Ray, Optuna, W&B, or MLflow.
- Does NOT build a frontend or dashboard.
- Does NOT make Gemini/API calls.

These come in future phases.
