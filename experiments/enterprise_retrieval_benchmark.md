# Enterprise Retrieval Benchmark

> **Status:** Phase 3B.2 — Realistic Enterprise Retrieval Benchmark
> **Last updated:** 2026-08-24

---

## Why the Original 3-Document Corpus Was Insufficient

The original controlled-retrieval pilot ran against a **3-document corpus**
(`data/machine_learning_intro.md`, `data/supervised_learning.md`,
`data/overfitting.md`) with an 8-question benchmark
(`data/retrieval_benchmark_questions.json`).

The pilot exposed a fundamental problem: the corpus was too small to
**distinguish** retrieval configurations. Every one of the four
configurations produced identical metrics:

| Configuration        | Recall  | Precision | F1   |
| -------------------- | ------- | --------- | ---- |
| A (chunk=100, top_k=3) | 1.0   | 0.3333    | 0.5  |
| B (chunk=200, top_k=3) | 1.0   | 0.3333    | 0.5  |
| C (chunk=300, top_k=5) | 1.0   | 0.3333    | 0.5  |
| D (chunk=500, top_k=5) | 1.0   | 0.3333    | 0.5  |

With only three documents, the retriever has almost nothing to be wrong
about: most questions retrieve the one relevant document (Recall = 1.0),
and the small pool of chunks makes Precision collapse to the same value
for every configuration (the correct document is 1 of 3 retrieved slots).

**A 3-document corpus cannot measure retrieval quality.** We will not
optimize against this toy corpus.

## Why a Larger Corpus Is Necessary

Retrieval quality is only meaningful when the retriever faces **distractor
content**:

- **Precision** needs irrelevant documents that compete for the top-k
  slots, so that a poor configuration actually returns the wrong
  documents.
- **Recall** needs a corpus large enough that relevant documents are not
  trivially guaranteed to appear in the top-k.
- **Chunking differences** only show up when documents are long and varied
  enough that chunk boundaries affect which fragment ends up in the top-k.
  A 3-document corpus chunks into so few pieces that chunk_size and
  chunk_overlap have hardly any observable effect.
- **Latency** differences only appear across enough documents and chunks to
  make the vector store size meaningful.

The enterprise knowledge base provides realistic volume, a controlled
amount of overlapping vocabulary across categories, and a strong distractor
structure — all without leaving the machine.

**"Retrieval configurations must be evaluated against a corpus that can
actually punish bad retrieval."**

---

## Enterprise Knowledge-Base Structure

The corpus lives in `data/enterprise_kb/`:

```
data/enterprise_kb/
├── hr/          (5 documents)
├── engineering/ (5 documents)
├── finance/     (5 documents)
├── product/     (5 documents)
└── security/    (5 documents)
```

**25 documents total**, each ~300-600 words, deterministic Markdown written
by `scripts/gen_kb_part1.py` and `scripts/gen_kb_part2.py`. Contents are fully
synthetic and contain distinct, verifiable facts (limits, thresholds,
approvals, teams, time periods, exceptions).

"The enterprise corpus is synthetic and exists solely for controlled
evaluation."

---

## Document Categories

| Category     | Documents                                                                  |
| ------------ | -------------------------------------------------------------------------- |
| HR           | leave_policy, remote_work_policy, employee_benefits, parental_leave, attendance_policy |
| Engineering  | deployment_guidelines, api_standards, coding_standards, incident_management, testing_guidelines |
| Finance      | travel_policy, expense_policy, reimbursement_policy, procurement_policy, budget_policy |
| Product      | product_roadmap, release_process, pricing_policy, feature_request_process, product_metrics |
| Security     | password_policy, access_control, security_incidents, data_classification, device_security |

Each document contains **distinct facts** (not generic text) so that a
question supported by one document is not equivalently supported by
another. For example, `travel_policy.md` contains:

> "Domestic hotel reimbursement is capped at 4,000 rupees per night for
> standard employees and 6,000 rupees per night for senior managers."

...while `expense_policy.md` contains unrelated, non-overlapping limits,
such as the 30-day submission deadline and the director/CFO approval
bands. Answer duplication across documents is deliberately avoided.
---

## Question Construction

`data/enterprise_retrieval_questions.json` contains **20 questions**. Every
question:

- Has a unique `case_id` (`ent-001` … `ent-020`).
- Requires a **specific factual** answer that exists in the enterprise KB
  (limits, durations, approval thresholds, teams).
- Sets `expected_action = "SEARCH"` (this is a forced-retrieval benchmark;
  the decision layer is bypassed).
- Lists exactly the `relevant_documents` needed to answer it.
- Carries `expected_answer` derived verbatim from the document content.

Selection is **deterministic**: running the pilot with `--max-questions 2`
always uses the first two cases (`ent-001`, `ent-002`) for **every**
configuration.

Examples:

- "How many days of paid annual leave does a full-time employee receive
  per year?" → `hr/leave_policy.md` (24 days).
- "What is the domestic hotel reimbursement cap per night for standard
  employees?" → `finance/travel_policy.md` (4,000 rupees).
- "Who must approve every production deployment?" →
  `engineering/deployment_guidelines.md` (tech lead + on-call SRE).

## Distractor Documents

Each question has **1 relevant document plus several irrelevant documents
containing related vocabulary**, so a retriever cannot succeed merely by
matching generic enterprise words.

Example — "What is the domestic hotel reimbursement cap per night for
standard employees?"

- **Relevant:** `finance/travel_policy.md`
- **Distractors:** `finance/expense_policy.md`, `hr/employee_benefits.md`,
  `product/pricing_policy.md`

All three distractors contain money/limit/approval vocabulary, but only
`travel_policy.md` contains the hotel cap. This forces the retriever to
select on **specific** facts rather than keyword overlap.

## Multi-Document Questions

Three questions require information from **two documents**:

| Case    | Question requires                                       | Documents |
| ------- | ------------------------------------------------------- | --------- |
| ent-018 | Senior-manager hotel cap **and** director approval band | travel_policy, expense_policy |
| ent-019 | Password expiry **and** personal-device enrollment      | password_policy, device_security |
| ent-020 | Annual-contract discount **and** release cadence        | pricing_policy, release_process |

These make **recall** meaningful: with `top_k=3` and two relevant documents
(sometimes from different categories), a sub-optimal configuration can
legitimately return only one of the two required documents (Recall = 0.5),
while the retriever must still fight distractors.

---

## Evaluation Metrics

For every configuration the experiment records:

| Metric              | Role      | Source |
| ------------------- | --------- | ------ |
| retrieval precision | secondary | retrieval evaluator |
| retrieval recall    | secondary | retrieval evaluator |
| **retrieval F1**    | **primary** | retrieval evaluator |
| context relevance   | secondary | context evaluator |
| answer score        | optional  | only if generation runs |
| retrieval latency   | secondary | adapter timing |
| generation latency  | secondary | adapter timing (None if no generation) |
| total latency       | secondary | retrieval + generation |
| API errors          | tracked   | evaluation status |

Because this is a **forced-retrieval** benchmark, agent decision accuracy is
deliberately **not** a primary metric — the agent is never asked to decide.
---

## Controlled Configurations

The four existing configurations are unchanged (`experiments/adaptiverag_configs.json`):

| Config | chunk_size | chunk_overlap | top_k |
| ------ | ---------- | ------------- | ----- |
| A      | 100        | 20            | 3     |
| B      | 200        | 40            | 3     |
| C      | 300        | 50            | 5     |
| D      | 500        | 75            | 5     |

No values are optimized against the enterprise KB. No automatic search is
performed — the configurations are fixed control points.

---

## Pipeline: Forced Retrieval Only

Retrieval mode uses `retrieval_benchmark_adapter.py` with
`enable_generation=False`, so the pilot makes **no Gemini API calls**:

- The adapter loads `data/enterprise_kb/` (recursively) into a vector store
  per configuration.
- Every question is a forced SEARCH; the agent decision layer is bypassed.
- Retrieval metrics, context relevance, and latency are computed locally.
- If a configuration fails, the failure is recorded, metrics are marked as
  errors, and the next configuration continues. Nothing is fabricated.

---

## Limitations

- **Synthetic data**: the corpus does not represent real company policy;
  it only realistically *resembles* an enterprise knowledge base in size
  and structure.
- **Small scale**: 25 documents is realistic enough to test retrieval
  mechanics but is not a production corpus.
- **Pilot scale**: the pilot numbers below are a *pilot*, not a conclusion;
  2 questions × 4 configurations keeps the loop tight.
- **Single embedding heuristic**: the local TF-IDF-style vector store is
  deterministic and offline, which is ideal for benchmarking but not a
  proxy for an industrial embedding model.
- **Synthetic distractor structure** is engineered by hand; real corpora
  have less controllable ambiguity.

## Why This Is Still a Synthetic Benchmark

The entire corpus is machine-generated by deterministic scripts, with
synthetic rupee figures, synthetic thresholds, and synthetic teams. It is
not a scrape of any website, not copyrighted material, and not private
company data. It exists to make retrieval evaluation meaningful,
repeatable, and free of external dependencies.

"The enterprise corpus is synthetic and exists solely for controlled
evaluation."

## Why This Is Experimentation Rather Than Optimization

The goal of this phase is to **measure** the four fixed configurations on a
realistically ambiguous corpus, identify *whether* they can be
distinguished, and decide the right dimensions (chunking, top-k, distractor
structure) for future experiments. We deliberately do not, yet, search for
optimal values, add auto-tuning, hook up Optuna/Ray/MLflow, or modify the
reward/evaluation methodology. Optimization begins only after the
benchmark demonstrates discriminative power.

---

## Pilot Results

Two-question pilot (`--mode retrieval --dataset enterprise --max-questions 2
--delay 5`): the same 2 questions (`ent-001`, `ent-002`) were used for all
four configurations. Generation was disabled (no Gemini calls were made), so
answer scores are not reported here.

| Metric              | Config A | Config B | Config C | Config D |
| ------------------- | -------- | -------- | -------- | -------- |
| chunk_size          | 100      | 200      | 300      | 500      |
| chunk_overlap       | 20       | 40       | 50       | 75       |
| top_k               | 3        | 3        | 5        | 5        |
| retrieval precision | 0.7500   | 0.4167   | 0.2917   | 0.2000   |
| retrieval recall    | 1.0000   | 1.0000   | 1.0000   | 1.0000   |
| **retrieval F1**    | **0.8333** | **0.5833** | **0.4500** | **0.3333** |
| context relevance   | 0.7670   | 0.8920   | 0.8920   | 0.8920   |
| retrieval latency   | 79.4 ms  | 42.3 ms  | 33.9 ms  | 20.8 ms  |
| API errors          | 0        | 0        | 0        | 0        |

Observations (pilot only, 2 questions — not a conclusion):

- **Recall is still saturated at 1.0** for this 2-question sample, but
  **precision and F1 now vary meaningfully** across configurations
  (F1 from 0.3333 to 0.8333), which the 3-document corpus could not do.
- Config A (chunk=100, top_k=3) achieved the best retrieval F1 and
  precision; Config D was the fastest.
- These numbers are a pilot sanity check, not an optimization result;
  the full 20-question benchmark is required before any configuration is
  preferred.

The saved artifacts are `experiments/results/adaptiverag_retrieval_enterprise_*.json`
(local, git-ignored) and are not part of the committed methodology.