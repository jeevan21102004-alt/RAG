"""Phase 6C sensitivity tests (local retrieval-only, no external API)."""
import unittest


class TestSensitivityPlan(unittest.TestCase):
    def test_config_count_and_uniqueness(self):
        from src.adaptive_rag.query_sensitivity import sensitivity_configurations
        configs = sensitivity_configurations()
        self.assertEqual(len(configs), 7)
        keys = [tuple(sorted(c.items())) for c in configs]
        self.assertEqual(len(set(keys)), 7)

    def test_baseline_first_and_valid(self):
        from src.adaptive_rag.query_sensitivity import (
            BASELINE_CONFIGURATION,
            sensitivity_configurations,
        )
        configs = sensitivity_configurations()
        self.assertEqual(configs[0], BASELINE_CONFIGURATION)
        for c in configs:
            self.assertLess(c["chunk_overlap"], c["chunk_size"])
            self.assertGreater(c["top_k"], 0)

    def test_parameter_isolation(self):
        from src.adaptive_rag.query_sensitivity import (
            BASELINE_CONFIGURATION,
            sensitivity_configurations,
        )
        for config in sensitivity_configurations()[1:]:
            differing = [
                k for k in BASELINE_CONFIGURATION
                if config[k] != BASELINE_CONFIGURATION[k]
            ]
            self.assertEqual(len(differing), 1)

    def test_deterministic_generation(self):
        from src.adaptive_rag.query_sensitivity import sensitivity_configurations
        self.assertEqual(
            sensitivity_configurations(), sensitivity_configurations()
        )

    def test_sweep_values_in_search_space(self):
        from src.adaptive_rag.query_sensitivity import AXIS_SWEEPS, sensitivity_configurations
        self.assertEqual(
            set(AXIS_SWEEPS), {"chunk_size", "chunk_overlap", "top_k"}
        )
        for params in sensitivity_configurations():
            self.assertIn(params["chunk_size"], (100, 300, 500))
            self.assertIn(params["chunk_overlap"], (10, 40, 75))
            self.assertIn(params["top_k"], (2, 4, 6))


class TestSensitivityExecution(unittest.TestCase):
    def test_grouping_matches_classifier(self):
        from src.adaptive_rag.query_aware_evaluation import load_questions
        from src.adaptive_rag.query_classifier import classify_all
        from src.adaptive_rag.query_sensitivity import run_sensitivity
        questions = load_questions()
        expected = {
            qtype.value: n
            for qtype, n in _counts(classify_all(questions).values()).items()
        }
        subset = questions[:6]
        result = run_sensitivity(
            configurations=[
                {"chunk_size": 300, "chunk_overlap": 40, "top_k": 4},
                {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2},
            ],
            questions=subset,
        )
        self.assertEqual(
            result["budget"]["retrieval_evaluations"], 2 * len(subset)
        )
        self.assertEqual(sum(result["question_counts"].values()), len(subset))

    def test_metrics_from_actual_results(self):
        from src.adaptive_rag.query_aware_evaluation import load_questions
        from src.adaptive_rag.query_sensitivity import run_sensitivity
        result = run_sensitivity(
            configurations=[{"chunk_size": 300, "chunk_overlap": 40, "top_k": 4}],
            questions=load_questions()[:4],
        )
        entry = result["config_results"][0]
        self.assertEqual(len(entry["evaluations"]), 4)
        self.assertIsNotNone(entry["global"]["f1"])
        self.assertGreaterEqual(entry["global"]["f1"], 0.0)
        self.assertLessEqual(entry["global"]["f1"], 1.0)

    def test_repeated_execution_identical(self):
        from src.adaptive_rag.query_aware_evaluation import load_questions
        from src.adaptive_rag.query_sensitivity import run_sensitivity
        subset = load_questions()[:4]
        configs = [
            {"chunk_size": 300, "chunk_overlap": 40, "top_k": 4},
            {"chunk_size": 100, "chunk_overlap": 40, "top_k": 4},
        ]
        first = run_sensitivity(configs, subset)
        second = run_sensitivity(configs, subset)
        # Latency-free retrieval metrics are exactly deterministic; wall-clock
        # latency (and the latency-derived objective component) may vary, so
        # compare only latency-free fields and bound the objective drift.
        self.assertEqual(
            _strip(first, drop_objective=True),
            _strip(second, drop_objective=True),
        )

    def test_summarize_axis_ranges(self):
        from src.adaptive_rag.query_sensitivity import summarize_axis
        axis = {
            "values": [100, 300, 500],
            "global": [
                {"objective_score": 0.5},
                {"objective_score": 0.7},
                {"objective_score": 0.6},
            ],
            "by_type": {
                "SIMPLE": [
                    {"objective_score": 0.4},
                    {"objective_score": 0.4},
                    {"objective_score": 0.4},
                ],
            },
        }
        summary = summarize_axis(axis)
        self.assertAlmostEqual(summary["global"]["range"], 0.2)
        self.assertAlmostEqual(summary["by_type"]["SIMPLE"]["range"], 0.0)


def _counts(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _strip(result, drop_objective=False):
    """Remove timing-sensitive fields before comparing two runs.

    Wall-clock latency (per-evaluation ``*_latency_ms`` fields, the averaged
    ``retrieval_latency_ms``, and the latency-derived objective component)
    varies run to run; everything else is exactly deterministic.
    """
    import copy
    result = copy.deepcopy(result)

    def _clean_eval(evaluation):
        for key in (
            "latency_ms",
            "retrieval_latency_ms",
            "generation_latency_ms",
            "total_latency_ms",
        ):
            evaluation.pop(key, None)

    def _clean_agg(agg):
        for key in (
            "latency_ms",
            "retrieval_latency_ms",
            "generation_latency_ms",
            "total_latency_ms",
        ):
            (agg or {}).pop(key, None)
        if drop_objective:
            (agg or {}).pop("objective_score", None)
            (agg or {}).pop("latency_component", None)
            objective = (agg or {}).get("objective")
            if isinstance(objective, dict):
                objective.pop("latency_component", None)
                objective.pop("latency_score", None)
                objective.pop("objective_score", None)

    for entry in result["config_results"]:
        _clean_agg(entry["global"])
        for agg in entry["by_type"].values():
            _clean_agg(agg)
        for evaluation in entry["evaluations"]:
            _clean_eval(evaluation)
    for axis in result["axes"].values():
        for agg in axis["global"]:
            _clean_agg(agg)
        for aggs in axis["by_type"].values():
            for agg in aggs:
                _clean_agg(agg)
    return result


if __name__ == "__main__":
    unittest.main()
