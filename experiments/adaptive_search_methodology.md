# Adaptive Search Methodology (Phase 3D)

**Status:** Phase 3D — Deterministic Adaptive Configuration Search
**Prior phase:** 3C (exhaustive grid search, commit `5dcd4bb`)
**Implementation:** `src/adaptive_rag/adaptive_search.py`

> **This is a deterministic heuristic adaptive search strategy.**
>
> **It is NOT Bayesian optimization, reinforcement learning, or a guarantee
> of global optimality.**

No PPO, GRPO, Q-learning, policy gradients, neural networks, Bayesian
optimization libraries, Optuna, Ray, or evolutionary algorithms are used.
Only the Python standard library and existing project dependencies.

---

## 1. Why Grid Search Becomes Expensive

Phase 3C enumerates every configuration in the search space. The cost is
multiplicative:

| Space | Combinations | Evaluations (2 questions) |
|-------|--------------|---------------------------|
| 3C pilot | 2×2×2 = 8 | 16 |
| 3D pilot | 7×5×5 = **175** | 350 |
| full | larger still | prohibitive |

Evaluating all 175 configurations for a marginal quality gain is wasteful:
most of the budget is spent on clearly inferior regions. An adaptive
strategy spends the budget where evidence suggests promising regions are.

## 2. Adaptive Search Concept

```
GRID:      generate all -> evaluate all -> rank
ADAPTIVE:  generate candidates -> evaluate some -> observe results ->
           select promising neighbour -> evaluate -> repeat (budget)
```

The engine decides **WHAT TO RUN NEXT**; how experiments run is delegated
entirely to the existing `ExperimentRunner`, `RetrievalBenchmarkAdapter`,
`EvaluationRunner`, `RetrievalEvaluator`, diagnostic engine, and objective
function. No evaluation logic is duplicated.

## 3. Initial Exploration Strategy

Before any adaptation is possible, the engine evaluates a small,
deterministic set spread across the space (function
`select_initial_combinations`):

1. **min-min-min corner** — smallest `chunk_size`, smallest
   `chunk_overlap`, smallest `top_k`.
2. **max-max-max corner** — largest of all three parameters.
3. **opposite corner A** — max `chunk_size`, min `chunk_overlap`,
   min `top_k`.
4. **opposite corner B** — min `chunk_size`, max `chunk_overlap`,
   max `top_k`.
5. **centre** — middle value of each parameter list
   (`values[len(values)//2]`).

Duplicates are removed preserving order. For the 175-combination pilot
space this yields exactly five configurations:

| # | chunk_size | chunk_overlap | top_k |
|---|-----------|---------------|-------|
| 1 | 100 | 10 | 2 |
| 2 | 500 | 75 | 6 |
| 3 | 500 | 10 | 2 |
| 4 | 100 | 75 | 6 |
| 5 | 250 | 40 | 4 |

The selection is a pure function of the sorted parameter lists — no
randomness, so repeated runs select identical configurations in identical
order.

## 4. Neighbour Definition

A **neighbour** differs by exactly ONE parameter step: one parameter moves
one position up or down its sorted value list; all others are unchanged
(function `get_neighbors`). Neighbours of `(250, 40, 4)`:

```
(200, 40, 4)  (300, 40, 4)      <- chunk_size steps
(250, 20, 4)  (250, 50, 4)      <- chunk_overlap steps
(250, 40, 3)  (250, 40, 5)      <- top_k steps
```

Only valid combinations are produced (stepping outside a list yields no
neighbour; space validation guarantees `chunk_overlap < chunk_size`).

## 5. Neighbor Scoring

Every unevaluated candidate receives a deterministic priority:

```
priority(c) = neighbour_average_objective(c) + exploration_bonus
              if c has >= 1 evaluated neighbour with an objective

priority(c) = exploration_bonus
              if c has no evaluated neighbour (or none has an objective)
```

- `neighbour_average_objective(c)` averages objective scores of already
  evaluated neighbours (failed neighbours excluded).
- The formula is transparent and printed in each entry's
  `selection_reason`.

This is explicitly a heuristic, not a mathematically optimal optimizer.

## 6. Exploration vs. Exploitation

- **Exploitation:** candidates near well-scoring configurations get high
  priority (their neighbour average dominates).
- **Exploration:** candidates in untouched regions get exactly the
  exploration bonus.

`exploration_bonus` defaults to **0.05** (CLI: `--exploration-bonus`).
Effect: a candidate whose evaluated neighbours average below ~0.05 loses
to any unexplored candidate; better regions keep priority. A small bonus
keeps the search mostly exploitative after the initial phase while still
guaranteeing unexplored areas remain reachable.

## 7. Selection Process

Each adaptive step scans every unevaluated combination, computes its
priority, and selects the maximum. Ties break on the deterministic
parameter key ascending, so repeated runs select identical sequences.
The loop stops at the `max_configurations` budget or full exhaustion.

## 8. Search Efficiency

Defined identically for both strategies (`search_comparison.py`):

```
search_efficiency = best_objective_found / configurations_evaluated
space_coverage    = configurations_evaluated / total_possible_configurations
```

Example: 8 of 175 evaluated -> coverage 4.57%. The comparison reports
measured values only and does not claim either strategy is superior.

## 9. Determinism

- Initial exploration: pure function of parameter lists.
- Priorities: stored objectives + constant bonus.
- Tie-breaking: fixed parameter-key ordering.
- Questions: first N cases of the dataset JSON.
- Tests: deterministic mock adapter (no API calls).

Wall-clock latency varies naturally between runs and can perturb the
latency term of the objective; ranking tests therefore use deterministic
mock latency. This documented limitation applies here as well.

## 10. Failure Handling

- **CONFIGURATION FAILURE** — one evaluation raises: recorded as
  `CONFIG_ERROR: ...` with `objective_score = None`, excluded from
  neighbour averaging, search continues.
- **SYSTEM FAILURE** — missing dataset/corpus or zero loadable cases:
  raises `SearchSystemError`, run stops.
- Failed values are never fabricated as metrics.

## 11. Limitations

1. **Heuristic, not optimal** — neighbour-average scoring can miss optima
   separated from good regions by ridges; no convergence proof.
2. **Greedy** — no lookahead or memory beyond immediate neighbours.
3. **Tiny pilot** — 8/175 configurations validate the mechanism only;
   they cannot establish superiority over grid search.
4. **Latency variability** — see §9.
5. **Answer quality excluded** — generation stays disabled; only F1,
   context relevance, and latency feed the objective.
6. **Single dataset** — findings need re-validation on other corpora
   before generalizing.