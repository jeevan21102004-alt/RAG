# AdaptiveRAG

An evaluation, diagnosis, and optimization framework for enterprise RAG retrieval systems.

## Problem

Simply building a RAG chatbot is insufficient because:

- Retrieval quality can fail: the wrong documents (or wrong chunks) are returned.
- Retrieved context can be irrelevant even when it shares keywords with the question.
- Answers can be poorly grounded in the retrieved context.
- Retrieval configuration (chunk size, chunk overlap, top-k) directly affects quality and latency.
- Optimization requires measurable experiments, not guesswork.

AdaptiveRAG addresses this by making retrieval quality measurable, diagnosable, and optimizable.

## What AdaptiveRAG does

The pipeline is:

```text
Enterprise Knowledge Base
  -> Retrieval
  -> Evaluation
  -> Failure Diagnosis
  -> Experimentation
  -> Configuration Search
  -> RL-based Configuration Selection
  -> Interactive Dashboard
```

Given an enterprise corpus and benchmark questions, the framework runs
controlled retrieval experiments, scores them with retrieval metrics,
diagnoses failures, searches the configuration space for better settings,
and presents everything in an interactive dashboard.

## Architecture

Major modules under `src/adaptive_rag/`:

- `evaluation_schema.py` — unified evaluation schema (cases, responses, results).
- `answer_evaluator.py` — answer quality evaluation.
- `retrieval_evaluator.py` — retrieval metrics: precision, recall, F1, context relevance, latency.
- `diagnostics.py` — failure-diagnosis engine (`diagnose_result()` / `diagnose_system()`).
- `experiment_config.py`, `experiment_runner.py`, `experiment_storage.py`, `experiment_comparison.py` — experiment framework.
- `retrieval_benchmark_adapter.py` — forced-retrieval adapter; retrieval-only runs with generation disabled (zero API calls).
- `search_space.py`, `config_generator.py`, `grid_search.py` — Phase 3C deterministic grid search.
- `adaptive_search.py`, `adaptive_search_result.py` — Phase 3D deterministic adaptive (neighbor-heuristic) search.
- `optimization_state.py`, `optimization_reward.py`, `rl_optimization_environment.py`, `configuration_policy.py`, `optimization_training.py`, `optimization_baselines.py`, `optimization_result.py` — Phase 4 RL configuration selection.
- `dashboard_service.py` — thin, API-free service layer for the dashboard; no evaluation logic of its own.
- `app/dashboard.py` — Streamlit dashboard (Demo Mode, no external API calls).

## Enterprise benchmark

Existing facts about the committed benchmark:

- 25 synthetic enterprise documents in `data/enterprise_kb/` (5 categories: HR, Engineering, Finance, Product, Security; 5 documents each).
- 20 benchmark questions in `data/enterprise_retrieval_questions.json`.
- 3 multi-document questions (answers requiring two documents).
- Retrieval-only experiments run locally with generation disabled: zero Gemini calls, no internet access.

## Optimization

- **Phase 3C — grid search:** exhaustive, deterministic search over a manually defined parameter space.
- **Phase 3D — deterministic adaptive search:** transparent neighbor-based heuristic using previous results (not Bayesian optimization, not RL).
- **Phase 4 — RL configuration selection:** small REINFORCE policy observing optimization history and choosing the next configuration ID under a limited budget.
- **Search space:** 175 configurations (chunk sizes `[100, 150, 200, 250, 300, 400, 500]`, overlaps `[10, 20, 40, 50, 75]`, top-k `[2, 3, 4, 5, 6]`).

The RL smoke test did not outperform the simpler baselines.

## Phase 4 measured comparison

Phase 4 smoke-test comparison, budget=4, seed=42.

| Method   | Best objective | Best F1 | Context relevance | Latency (ms) |
|----------|---------------|---------|-------------------|--------------|
| Random   | 0.7434        | 0.8333  | 0.8295            | 22.44        |
| Grid     | 0.7419        | 0.8333  | 0.8295            | 85.88        |
| Adaptive | 0.7434        | 0.8333  | 0.8295            | 22.44        |
| RL       | 0.6441        | 0.5833  | 0.8295            | 85.92        |

Random and Adaptive tied for the best measured objective (0.7434). RL scored
lowest (0.6441) after only 5 training episodes — an exploratory smoke-test
result, not evidence about RL in general.

## Dashboard

- Built with Streamlit (`app/dashboard.py`): `python -m streamlit run app/dashboard.py`.
- 7 sections: Dashboard, Knowledge Base, Retrieval Evaluation, Failure Diagnosis, Experiments, Optimizer, Configuration Explorer.
- Demo Mode only: zero external API calls, no Gemini key required.
- Uses the local enterprise corpus, live local retrieval evaluation (generation disabled), labelled saved experiment records, and explicit "Not evaluated" labels for unevaluated configurations.
- Includes a failure-diagnosis view (existing engine) and an optimizer view showing the measured recommendation with its source.


## Installation

Windows-friendly setup from the repository root:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run dashboard

```bat
python -m streamlit run app/dashboard.py
```

The dashboard runs in Demo Mode and does not require a Gemini API key.

## Testing

Currently verified dashboard tests:

```bat
python -m unittest discover -s tests -p test_dashboard_service.py
```

9 dashboard tests currently verified.

The 321-test figure from the prior project baseline refers to the previously
verified full suite, not a freshly re-run result in this documentation phase.

## Repository structure

```text
app/                  Streamlit dashboard (dashboard.py)
src/adaptive_rag/     Evaluation, diagnostics, experiments, search, RL, dashboard service
data/                 Enterprise corpus (enterprise_kb/), benchmark questions, RL split
tests/                Unit tests, including test_dashboard_service.py
scripts/              Dataset generation and validation utilities
models/               Saved policies and experiment metadata
requirements.txt      Python dependencies
```

## Reproducibility

- Grid search, adaptive search, and search-space enumeration are deterministic:
  the same search space always yields the same configurations in the same order.
- The Phase 4 comparison values above are saved/historical smoke-test records
  (budget=4, seed=42), not live computations.
- Dashboard evaluations that do run live are single-question, retrieval-only,
  local computations with generation disabled.
- Unevaluated configurations are explicitly shown as "Not evaluated" — no
  metrics are fabricated.

## Limitations

- Answer evaluation is heuristic, not a human or strong LLM judge.
- The enterprise benchmark is synthetic and small (25 documents, 20 questions).
- The Phase 4 comparison used a small smoke-test budget (4 evaluations per
  method, 5 RL training episodes), so results are exploratory.
- The RL result is exploratory; no claim of global optimality is made for any method.
- The dashboard is currently retrieval/demo focused.
- Full production deployment is outside the project scope.

## Roadmap

- Larger real-world benchmark datasets.
- Stronger evaluation judges.
- Broader optimization budgets.
- Production integrations.
- Optional live generation mode (explicitly opt-in, keyed).

## Baseline RAG pipeline (original)

The repository grew out of a simple baseline RAG pipeline, which is still
present:

- `src/adaptive_rag/data_loader.py` reads the sample files in `data/`.
- `src/adaptive_rag/chunking.py` splits each document into small overlapping text chunks.
- `src/adaptive_rag/embeddings.py` turns the chunks into word-frequency embedding vectors.
- `src/adaptive_rag/vector_store.py` stores those vectors and the chunk text in `storage/vector_store.json`.
- `src/adaptive_rag/retrieval.py` embeds the user question, computes cosine similarity, and returns the best matches.
- `src/adaptive_rag/app.py` wires the pieces together.
- `main.py` runs a baseline query, e.g. `python main.py --query "What is machine learning?"`.

The code is split into small modules so each piece can be improved without
rewriting the whole project.
