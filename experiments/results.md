# AdaptiveRAG RL Experiments

## Experiment 1 — 5-Feature REINFORCE (Asymmetric Reward)

- **State**: 5 handcrafted question features (question_length, word_count, question_mark_count, keyword_hits, document_reference)
- **Reward**: asymmetric (+2 correct, -1 unnecessary SEARCH, -3 missed SEARCH)
- **Algorithm**: REINFORCE
- **Seed**: 42
- **Episodes**: 2000
- **Train/Test split**: 14 train / 6 test (deterministic, seed 42)

### Result

| Metric | Value |
|--------|-------|
| Test Accuracy | 66.67% |
| SEARCH Accuracy | 100.00% |
| ANSWER Accuracy | 0.00% |
| Average Reward | 1.000 |
| SEARCH Rate | 100.00% |
| ANSWER Rate | 0.00% |
| Majority Baseline | 66.67% |

### Why Experiment 1 Failed

The policy collapsed to always predicting SEARCH. This is a known REINFORCE failure mode:

1. **Asymmetric penalties**: The missed-SEARCH penalty (-3) is 3x worse than the unnecessary-SEARCH penalty (-1). The policy learns that always SEARCHing is "safer" than risking the -3 penalty.
2. **Class imbalance**: The majority of questions (14/20) expect SEARCH, so always SEARCHing achieves 66.67% accuracy — matching the majority baseline.
3. **Limited state representation**: The 5 handcrafted features may not capture enough semantic signal to distinguish retrieval-required from retrieval-not-required questions.

## Experiment 2 — Semantic-State REINFORCE (Balanced Reward)

- **State**: 384-dimensional semantic embeddings (all-MiniLM-L6-v2)
- **Reward**: balanced (+1 correct, -1 unnecessary SEARCH, -1 missed SEARCH)
- **Algorithm**: REINFORCE
- **Seed**: 42
- **Episodes**: 2000
- **Train/Test split**: 14 train / 6 test (SAME split as Experiment 1)

### Result

| Metric | Value |
|--------|-------|
| Test Accuracy | 100.00% |
| SEARCH Accuracy | 100.00% |
| ANSWER Accuracy | 100.00% |
| Average Reward | 1.000 |
| SEARCH Rate | 66.67% |
| ANSWER Rate | 33.33% |
| Majority Baseline | 66.67% |

### Why Experiment 2 Improved

1. **Balanced reward**: Equal penalties (-1) for both unnecessary SEARCH and missed SEARCH remove the incentive to always SEARCH.
2. **Semantic state**: The 384-dimensional embeddings capture semantic meaning that the 5 handcrafted features miss, allowing the policy to distinguish retrieval-required from retrieval-not-required questions.
3. **No policy collapse**: The policy uses both actions (66.67% SEARCH, 33.33% ANSWER), matching the test set distribution.

## Comparison Table

| Experiment | State                  | Reward     | Test Accuracy | Majority Baseline | SEARCH Rate |
| ---------- | ---------------------- | ---------- | ------------- | ----------------- | ----------- |
| Exp 1      | 5 handcrafted features | asymmetric | 66.67%        | 66.67%            | 100%        |
| Exp 2      | semantic embeddings    | balanced   | 100.00%       | 66.67%            | 66.67%      |

## Conclusion

Semantic state representation + balanced reward **significantly improved** the RL retrieval policy:

- Test accuracy improved from 66.67% → 100.00%
- Experiment 2 beats the majority baseline (100% vs 66.67%)
- No policy collapse occurred in Experiment 2
- The balanced reward removed the incentive to always SEARCH
- The semantic embeddings provided richer state information than handcrafted features