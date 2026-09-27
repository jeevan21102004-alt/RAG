"""Phase 6D adaptive-depth tests (local retrieval-only, no external API)."""
import unittest


class TestDifficulty(unittest.TestCase):
    def test_low_by_default(self):
        from src.adaptive_rag.adaptive_retrieval import (
            RetrievalDifficulty, estimate_difficulty)
        self.assertIs(
            estimate_difficulty("What is the leave policy?"), RetrievalDifficulty.LOW)

    def test_high_multi_document(self):
        from src.adaptive_rag.adaptive_retrieval import (
            RetrievalDifficulty, estimate_difficulty)
        self.assertIs(
            estimate_difficulty("Compare A and B", relevant_documents=["a", "b"]),
            RetrievalDifficulty.HIGH)

    def test_medium_technical(self):
        from src.adaptive_rag.adaptive_retrieval import (
            RetrievalDifficulty, estimate_difficulty)
        self.assertIs(
            estimate_difficulty("What is the API latency p95?"),
            RetrievalDifficulty.MEDIUM)

    def test_deterministic(self):
        from src.adaptive_rag.adaptive_retrieval import estimate_difficulty
        first = estimate_difficulty("What is the API latency p95?")
        second = estimate_difficulty("What is the API latency p95?")
        self.assertIs(first, second)


class TestPolicy(unittest.TestCase):
    def test_default_mapping(self):
        from src.adaptive_rag.adaptive_retrieval import (
            AdaptiveDepthPolicy, RetrievalDifficulty)
        policy = AdaptiveDepthPolicy()
        self.assertEqual(policy.top_k_for(RetrievalDifficulty.LOW), 2)
        self.assertEqual(policy.top_k_for(RetrievalDifficulty.MEDIUM), 3)
        self.assertEqual(policy.top_k_for(RetrievalDifficulty.HIGH), 4)

    def test_configurable(self):
        from src.adaptive_rag.adaptive_retrieval import (
            AdaptiveDepthPolicy, RetrievalDifficulty, select_top_k)
        policy = AdaptiveDepthPolicy(low_top_k=5, medium_top_k=5, high_top_k=5)
        self.assertEqual(select_top_k("plain question", policy=policy), 5)

    def test_select_top_k_deterministic(self):
        from src.adaptive_rag.adaptive_retrieval import select_top_k
        self.assertEqual(
            select_top_k("What is the leave policy?"),
            select_top_k("What is the leave policy?"))

    def test_explain_consistent(self):
        from src.adaptive_rag.adaptive_retrieval import (
            estimate_difficulty, explain_decision, select_top_k)
        question = "What is the API latency p95?"
        explanation = explain_decision(question)
        self.assertEqual(
            explanation["difficulty"], estimate_difficulty(question).value)
        self.assertEqual(explanation["top_k"], select_top_k(question))
        self.assertEqual(
            explanation["configuration"]["top_k"], select_top_k(question))
        self.assertTrue(explanation["rules_fired"])

    def test_all_benchmark_questions_decided(self):
        from src.adaptive_rag.adaptive_retrieval import (
            RetrievalDifficulty, decide_for_question)
        from src.adaptive_rag.query_aware_evaluation import load_questions
        for item in load_questions():
            decision = decide_for_question(item)
            self.assertIn(
                decision["difficulty"], [d.value for d in RetrievalDifficulty])
            self.assertIn(decision["top_k"], (2, 3, 4))


class TestPilot(unittest.TestCase):
    def test_adaptive_evaluation_deterministic(self):
        from src.adaptive_rag.adaptive_retrieval import evaluate_policies
        from src.adaptive_rag.query_aware_evaluation import load_questions
        subset = load_questions()[:4]
        first = _strip(evaluate_policies(questions=subset))
        second = _strip(evaluate_policies(questions=subset))
        self.assertEqual(first, second)

    def test_budget_counts(self):
        from src.adaptive_rag.adaptive_retrieval import evaluate_policies
        from src.adaptive_rag.query_aware_evaluation import load_questions
        result = evaluate_policies(questions=load_questions()[:4])
        needed = result["budget"]["distinct_top_k_values"]
        self.assertEqual(result["budget"]["retrieval_evaluations"],
                         len(needed) * 4)


def _strip(result):
    """Remove timing-sensitive fields before comparing two runs."""
    import copy
    result = copy.deepcopy(result)
    for section in ("fixed", "adaptive"):
        agg = result[section]["aggregated"]
        for key in ("latency_ms", "retrieval_latency_ms",
                    "generation_latency_ms", "total_latency_ms",
                    "objective_score", "latency_component"):
            agg.pop(key, None)
        objective = agg.get("objective")
        if isinstance(objective, dict):
            for key in ("objective_score", "latency_score",
                        "latency_component"):
                objective.pop(key, None)
        for agg in result[section].get("by_type", {}).values():
            for key in ("latency_ms", "retrieval_latency_ms",
                        "generation_latency_ms", "total_latency_ms",
                        "objective_score", "latency_component"):
                (agg or {}).pop(key, None)
    for key in ("precision", "recall", "f1", "context_relevance",
                "retrieval_latency_ms", "objective_score"):
        result["delta_adaptive_minus_fixed"].pop(key, None)
    decisions = result.get("decisions", [])
    for decision in decisions:
        decision.pop("top_k", None)
        decision.pop("configuration", None)
    return result


if __name__ == "__main__":
    unittest.main()
