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
- `query_classifier.py`, `query_sensitivity.py`, `query_aware_evaluation.py` — Phase 6A/6B/6C deterministic query classification, sensitivity sweeps, and query-aware evaluation (no LLM).
- `adaptive_retrieval.py` — Phase 6D deterministic adaptive retrieval depth (per-query difficulty → top_k, no LLM/RL).
- `adaptive_ablation.py` — Phase 6E ablation: FIXED_2 vs FIXED_4 vs FIXED_6 vs ADAPTIVE (80 retrieval-only evaluations).
- `rag_service.py` — thin, Streamlit-independent service over the validated 300/40/top_k=2 config; `generation.py` — opt-in grounded generation reusing `llm.py`.
- `app/chat.py` — Streamlit chat UI (retrieval-only default, opt-in generation).

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

## Query-aware retrieval and adaptive depth (Phase 6)

Phase 6 asks a narrower question than configuration search: does choosing
`top_k` per query (adaptive depth) beat a single fixed `top_k`?

The deterministic, feature-based policy (`adaptive_retrieval.py`, no LLM, no
RL) maps LOW → top_k=2, MEDIUM → top_k=3, HIGH → top_k=4 over a fixed
chunk_size=300 / overlap=40 context. Measured on the 20-question enterprise
benchmark (12 SIMPLE, 5 TECHNICAL, 3 MULTI_DOCUMENT), retrieval-only, with a
labelled deterministic budget of 4 distinct top_k values {2,3,4,6} x 20
questions = 80 evaluations and zero API calls (`adaptive_ablation.py`,
Phase 6E):

| Policy | Precision | Recall | F1 | Context relevance | Objective |
|--------|-----------|--------|-----|-------------------|-----------|
| FIXED top_k=2 | 0.4750 | 0.8250 | 0.5917 | 0.7913 | 0.6818 |
| FIXED top_k=4 | 0.2583 | 0.8750 | 0.3950 | 0.8222 | 0.5886 |
| FIXED top_k=6 | 0.1858 | 0.9000 | 0.3045 | 0.8532 | 0.5357 |
| ADAPTIVE (2/3/4) | 0.4042 | 0.8750 | 0.5500 | 0.7976 | 0.6622 |

**Measured finding:** the fixed top_k=2 configuration is the best of the four
on precision, F1 and the objective. Adaptive depth clearly beats the wider
fixed settings (objective +0.0736 vs top_k=4, +0.1265 vs top_k=6) but does
**not** beat fixed top_k=2 (F1 -0.0417, objective -0.0197, recall +0.0500).
Raising top_k trades precision for recall — and slightly for context
relevance — on this corpus.

**What this result does not claim:** it is one synthetic 20-question
benchmark evaluated with a 3-tier heuristic difficulty estimate, so it is not
evidence that adaptive depth universally outperforms fixed depth — nor is the
opposite established beyond this corpus. The practical takeaway acted on is
the measured one: fixed top_k=2 at chunk_size=300 / overlap=40 is the
configuration the application uses.

## Dashboard

- Built with Streamlit (`app/dashboard.py`): `python -m streamlit run app/dashboard.py`.
- 7 sections: Dashboard, Knowledge Base, Retrieval Evaluation, Failure Diagnosis, Experiments, Optimizer, Configuration Explorer.
- Demo Mode only: zero external API calls, no Gemini key required.
- Uses the local enterprise corpus, live local retrieval evaluation (generation disabled), labelled saved experiment records, and explicit "Not evaluated" labels for unevaluated configurations.
- Includes a failure-diagnosis view (existing engine) and an optimizer view showing the measured recommendation with its source.


## Practical retrieval configuration

The application path uses one validated configuration instead of the search
space: **chunk_size=300 words, chunk_overlap=40 words, top_k=2** — the highest
measured F1 (0.5917) and objective (0.6818) in the Phase 6E ablation above.
It is defined once in `src/adaptive_rag/rag_service.py`
(`get_validated_config()`) and reused by the chat UI, so service, UI and
tests cannot drift apart.

## Practical chat application

```bat
python -m streamlit run app/chat.py
```

- `app/chat.py` is a thin UI over `RAGService`: the query goes through
  `query_rag` against the cached local corpus (`data/enterprise_kb/`), and
  the UI only formats the returned chunks, context, sources and retrieval
  metadata. There is no second retrieval pipeline.
- **Retrieval-only is the default**: no LLM, no network, no API key required.
- **Optional grounded generation is explicitly opt-in** via the UI toggle.
  When enabled, only the retrieved context is passed to
  `generate_grounded_answer` (`src/adaptive_rag/generation.py`), which reuses
  the existing Gemini integration in `llm.py`. The key is read from
  `GEMINI_API_KEY` inside `llm.py` (`.env`, gitignored) and is never shown in
  the UI or logs, and no model call happens unless the toggle is on.
- Generation failures never break retrieval: an empty context skips the
  model, and any model error is surfaced as a warning while the retrieved
  sources are still displayed.

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

Last measured offline run (no Gemini/API calls; generation tests inject a stub):
9 dashboard tests, plus 49 tests across the six query-aware-retrieval and
practical-app suites (adaptive retrieval 11, ablation 6, RAG service 8,
generation 9, chat 8, practical flow 7) — all passing.

The 321-test figure from the prior project baseline refers to the previously
verified full suite, not a freshly re-run result in this documentation phase.

## Repository structure

```text
app/                  Streamlit UIs: dashboard.py (evaluation demo), chat.py (practical chat)
src/adaptive_rag/     Evaluation, diagnostics, experiments, search, RL, query-aware retrieval, service + generation layers
data/                 Enterprise corpus (enterprise_kb/), benchmark questions, RL split
tests/                Unit tests (dashboard, adaptive retrieval/ablation, RAG service, generation, chat, practical flow)
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
- Adaptive depth was evaluated only against fixed top_k on a 20-question
  synthetic benchmark; no claim of universal outperformance is made either way.
- The dashboard is currently retrieval/demo focused.
- Full production deployment is outside the project scope.

## Roadmap

- Larger real-world benchmark datasets.
- Stronger evaluation judges.
- Broader optimization budgets.
- Production integrations.
- Groundedness evaluation of the opt-in generation path (human or strong-model judge).

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
