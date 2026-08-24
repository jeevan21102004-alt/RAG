# Automated Retrieval Configuration Search — Methodology (Phase 3C)

**Status:** Phase 3C — Deterministic Automated Grid Search
**Date:** 2025-01-23
**Commit:** 2dc1fd678de1413a26cf142e9ab7e352f25b0f89

---

## 1. Why Automated Search Is Necessary

After Phase 3B.2 demonstrated that the enterprise corpus discriminates
between retrieval configurations, manually testing four configurations by
hand becomes tedious and error-prone.  Automated search provides:

- **Reproducibility** — every configuration is generated, executed, and
  scored deterministically.
- **Scalability** — the search space can grow from 8 to hundreds of
  configurations without manual effort.
- **Traceability** — every configuration, metric, and score is recorded.

## 2. Experimentation vs. Optimization

This phase implements **experimental methodology**, not optimization:

- **Experimentation** — systematically evaluate a *finite set* of
  configurations and rank them by a transparent objective function.
- **Optimization** — adaptively *search* a space to find the global best,
  typically using Bayesian methods, evolutionary algorithms, or RL.

Phase 3C stops at experimentation: exhaustive enumeration.

> **This phase uses exhaustive grid search over a manually defined parameter
> space.**

> **No Bayesian optimization, reinforcement learning, evolutionary search,
> or neural optimization is used in Phase 3C.**

## 3. Grid Search Methodology

Grid search enumerates every combination of the discrete parameter values
defined in the `SearchSpace`:

```
for chunk_size in chunk_sizes:
    for chunk_overlap in overlaps:
        for top_k in top_ks:
            yield configuration
```

- Configurations are generated in a **deterministic order** (sorted
  ascending × ascending × ascending).
- Each configuration receives a **deterministic experiment ID** derived
  from its parameter hash.
- No randomness, no shuffling, no parallelization.

## 4. Search-Space Definition

The search space is defined in `src/adaptive_rag/search_space.py` as a
`SearchSpace` dataclass with:

| Parameter       | Pilot values           | Default values                              |
|-----------------|------------------------|---------------------------------------------|
| chunk_size      | `[100, 300]`           | `[100, 150, 200, 300, 400, 500]`            |
| chunk_overlap   | `[20, 50]`             | `[10, 20, 40, 50, 75]`                      |
| top_k           | `[3, 5]`               | `[2, 3, 4, 5, 6]`                           |

**Pilot:** 2 × 2 × 2 = **8 configurations**.

Validation rules enforced:

- All values must be positive integers.
- chunk_overlap < chunk_size (invalid pairs are rejected).
- Values must be sorted ascending for deterministic generation.

## 5. Configuration Generation

`src/adaptive_rag/config_generator.py` produces an ordered
`list[ExperimentConfig]`:

- Each config has a deterministic `experiment_id` (e.g., `grid_100_20_3`).
- Each config has a `name` and `description`.
- The same `SearchSpace` always yields the same configurations in the same
  order.

## 6. Objective Function

`src/adaptive_rag/objective.py` computes:

```
Objective = 0.50 × F1
          + 0.20 × ContextRelevance
          + 0.30 × LatencyScore
```

**Default weights:**

| Component        | Weight |
|------------------|--------|
| retrieval F1     | 0.50   |
| context relevance| 0.20   |
| latency score    | 0.30   |

## 7. Latency Normalization

Raw latency (milliseconds) is normalized to [0, 1] using:

```
latency_score = 1 / (1 + latency_ms / reference_latency_ms)
```

- `reference_latency_ms` defaults to **100 ms**.
- A configuration with 0 ms latency receives a perfect score of 1.0.
- A configuration with latency equal to the reference receives ≈0.5.
- Higher latency is penalized smoothly toward 0.

This prevents low-latency configurations from being unfairly penalized
when absolute latency is already very fast.

## 8. Failure Handling

The search distinguishes two failure classes:

| Type             | Example                                         | Action        |
|------------------|-------------------------------------------------|---------------|
| **CONFIGURATION FAILURE** | An individual configuration raises an error | Recorded as `CONFIG_ERROR`, search continues |
| **SYSTEM FAILURE**        | Dataset missing, corpus missing, or every configuration fails with the same error | `SearchSystemError` raised, search stops |

Failed configurations receive `objective_score = None` and are ranked
last.

## 9. Determinism

- Configuration generation is deterministic (sorted parameter values,
  hash-based IDs).
- Question selection is deterministic (first N cases from the JSON).
- The mock adapter returns deterministic results.
- **Latency may naturally vary** slightly between runs.  Ranking tests
  use mocked deterministic latency (`DeterministicMockAdapter`) so the
  ranking comparison is fully reproducible.

## 10. Pilot Design

The pilot uses:

- `--pilot` flag → 8 configurations (2 × 2 × 2).
- `--max-questions 2` → first 2 enterprise questions.
- Generation **disabled** → 0 Gemini API calls.
- Maximum 16 retrieval evaluations.

## 11. Limitations

1. **Exhaustive, not adaptive** — grid search scales poorly (exponential in
   parameter count).  Future phases may use smarter search strategies.
2. **Latency variability** — wall-clock latency naturally varies; the
   objective's latency component is therefore a proxy, not an exact
   measurement.  Tests use deterministic mock latency.
3. **No answer quality** — the objective includes only retrieval F1,
   context relevance, and latency.  Answer quality (Gemini-based) is
   deferred to a future phase.
4. **Synthetic corpus** — the enterprise KB is entirely synthetic.
   Results reflect retrieval mechanics, not real-world knowledge-base
   performance.
5. **Single corpus** — all configurations are evaluated against the same
   25-document synthetic corpus.
