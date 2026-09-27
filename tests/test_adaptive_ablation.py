"""Phase 6E ablation tests (local retrieval-only, no external API)."""
import unittest


class TestPolicies(unittest.TestCase):
    def test_four_policies(self):
        from src.adaptive_rag.adaptive_ablation import (
            POLICY_NAMES, describe_policies)
        self.assertEqual(list(POLICY_NAMES),
                         ["FIXED_2", "FIXED_4", "FIXED_6", "ADAPTIVE"])
        described = describe_policies()
        self.assertEqual(set(described), set(POLICY_NAMES))

    def test_fixed_selects_constant(self):
        from src.adaptive_rag.adaptive_ablation import top_k_for_policy
        for name, expected in (("FIXED_2", 2), ("FIXED_4", 4), ("FIXED_6", 6)):
            for decision in (None, {"top_k": 99}):
                self.assertEqual(top_k_for_policy(name, decision), expected)

    def test_adaptive_uses_6d_policy(self):
        from src.adaptive_rag.adaptive_ablation import top_k_for_policy
        from src.adaptive_rag.adaptive_retrieval import decide_for_question
        from src.adaptive_rag.query_aware_evaluation import load_questions
        for item in load_questions():
            expected = decide_for_question(item)["top_k"]
            self.assertEqual(
                top_k_for_policy("ADAPTIVE", decide_for_question(item)),
                expected)
        with self.assertRaises(ValueError):
            top_k_for_policy("ADAPTIVE", None)
        with self.assertRaises(ValueError):
            top_k_for_policy("UNKNOWN", None)


class TestAblation(unittest.TestCase):
    def test_deterministic(self):
        from src.adaptive_rag.adaptive_ablation import run_ablation
        from src.adaptive_rag.query_aware_evaluation import load_questions
        subset = load_questions()[:4]
        self.assertEqual(_strip(run_ablation(questions=subset)),
                         _strip(run_ablation(questions=subset)))

    def test_actual_results_and_comparison(self):
        from src.adaptive_rag.adaptive_ablation import (
            POLICY_NAMES, run_ablation)
        from src.adaptive_rag.query_aware_evaluation import load_questions
        result = run_ablation(questions=load_questions()[:6])
        self.assertEqual(set(result["results"]), set(POLICY_NAMES))
        for name in POLICY_NAMES:
            entry = result["results"][name]
            self.assertEqual(len(entry["evaluations"]), 6)
            self.assertIsNotNone(entry["aggregated"]["f1"])
        for key in ("ADAPTIVE_minus_FIXED_2", "ADAPTIVE_minus_FIXED_4",
                    "ADAPTIVE_minus_FIXED_6"):
            self.assertIn(key, result["differences"])
            diff = result["differences"][key]
            base = result["results"][key.replace("ADAPTIVE_minus_", "")]
            adaptive = result["results"]["ADAPTIVE"]
            self.assertAlmostEqual(
                diff["f1"],
                adaptive["aggregated"]["f1"] - base["aggregated"]["f1"])

    def test_budget(self):
        from src.adaptive_rag.adaptive_ablation import run_ablation
        from src.adaptive_rag.query_aware_evaluation import load_questions
        result = run_ablation(questions=load_questions()[:4])
        needed = result["budget"]["distinct_top_k_values"]
        self.assertEqual(result["budget"]["retrieval_evaluations"],
                         len(needed) * 4)


def _strip(result):
    """Remove timing-sensitive fields before comparing two runs."""
    import copy
    result = copy.deepcopy(result)
    for entry in result["results"].values():
        for key in ("latency_ms", "retrieval_latency_ms",
                    "generation_latency_ms", "total_latency_ms",
                    "objective_score", "latency_component",
                    "retrieval_component", "context_component"):
            entry["aggregated"].pop(key, None)
        objective = entry["aggregated"].get("objective")
        if isinstance(objective, dict):
            for key in ("objective_score", "latency_score",
                        "latency_component", "retrieval_component",
                        "context_component"):
                objective.pop(key, None)
        for agg in entry["by_type"].values():
            for key in ("latency_ms", "retrieval_latency_ms",
                        "generation_latency_ms", "total_latency_ms",
                        "objective_score", "latency_component",
                        "retrieval_component", "context_component"):
                (agg or {}).pop(key, None)
            objective = (agg or {}).get("objective")
            if isinstance(objective, dict):
                for key in ("objective_score", "latency_score",
                            "latency_component", "retrieval_component",
                            "context_component"):
                    objective.pop(key, None)
        for evaluation in entry["evaluations"]:
            for key in ("latency_ms", "retrieval_latency_ms",
                        "generation_latency_ms", "total_latency_ms"):
                evaluation.pop(key, None)
    for diff in result["differences"].values():
        for key in ("retrieval_latency_ms", "objective_score"):
            diff.pop(key, None)
    for key in ("precision", "recall", "f1", "context_relevance",
                "retrieval_latency_ms", "objective_score"):
        result.get("fixed2_reference", {}).pop(key, None)
    return result


if __name__ == "__main__":
    unittest.main()
