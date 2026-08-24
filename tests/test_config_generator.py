"""Tests for the deterministic configuration generator.

No API access, internet, or Gemini is required.
"""

import unittest

from src.adaptive_rag.config_generator import generate_configurations
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.search_space import SearchSpace


class TestNumberOfConfigurations(unittest.TestCase):
    def test_pilot_produces_eight_configs(self) -> None:
        configs = generate_configurations(SearchSpace.pilot(), prefix="grid")
        self.assertEqual(len(configs), 8)

    def test_default_space_count_matches(self) -> None:
        configs = generate_configurations(SearchSpace.default(), prefix="grid")
        self.assertEqual(len(configs), SearchSpace.default().count_combinations())


class TestDeterministicIds(unittest.TestCase):
    def test_ids_are_unique(self) -> None:
        configs = generate_configurations(SearchSpace.pilot(), prefix="grid")
        ids = [c.experiment_id for c in configs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ids_are_deterministic(self) -> None:
        space = SearchSpace.pilot()
        first = generate_configurations(space, prefix="grid")
        second = generate_configurations(space, prefix="grid")
        self.assertEqual(
            [c.experiment_id for c in first],
            [c.experiment_id for c in second],
        )

    def test_ids_reflect_ordering(self) -> None:
        configs = generate_configurations(SearchSpace.pilot(), prefix="grid")
        self.assertEqual(configs[0].experiment_id, "grid-1")
        self.assertEqual(configs[7].experiment_id, "grid-8")


class TestDeterministicOrdering(unittest.TestCase):
    def test_same_order_each_time(self) -> None:
        space = SearchSpace.pilot()
        a = [c.to_dict() for c in generate_configurations(space, prefix="grid")]
        b = [c.to_dict() for c in generate_configurations(space, prefix="grid")]
        self.assertEqual(a, b)

    def test_different_prefix_is_preserved(self) -> None:
        configs = generate_configurations(SearchSpace.pilot(), prefix="abc")
        self.assertTrue(configs[0].experiment_id.startswith("abc-"))


class TestParameterCombinations(unittest.TestCase):
    def test_every_config_has_full_parameter_set(self) -> None:
        for config in generate_configurations(SearchSpace.pilot(), prefix="grid"):
            self.assertIn("chunk_size", config.parameters)
            self.assertIn("chunk_overlap", config.parameters)
            self.assertIn("top_k", config.parameters)

    def test_all_combinations_appear(self) -> None:
        combos = {
            (c.parameters["chunk_size"], c.parameters["chunk_overlap"], c.parameters["top_k"])
            for c in generate_configurations(SearchSpace.pilot(), prefix="grid")
        }
        expected = {
            (100, 20, 3),
            (100, 20, 5),
            (100, 50, 3),
            (100, 50, 5),
            (300, 20, 3),
            (300, 20, 5),
            (300, 50, 3),
            (300, 50, 5),
        }
        self.assertEqual(combos, expected)

    def test_no_duplicate_combinations(self) -> None:
        combos = [
            (c.parameters["chunk_size"], c.parameters["chunk_overlap"], c.parameters["top_k"])
            for c in generate_configurations(SearchSpace.pilot(), prefix="grid")
        ]
        self.assertEqual(len(combos), len(set(combos)))

    def test_configs_are_experiment_config(self) -> None:
        for config in generate_configurations(SearchSpace.pilot(), prefix="grid"):
            self.assertIsInstance(config, ExperimentConfig)


if __name__ == "__main__":
    unittest.main()