"""Tests for the deterministic search space.

No API access, internet, or Gemini is required.
"""

import unittest

from src.adaptive_rag.search_space import SearchSpace


class TestValidSearchSpace(unittest.TestCase):
    def test_pilot_space_is_valid(self) -> None:
        space = SearchSpace.pilot()
        self.assertTrue(space.is_valid())
        self.assertEqual(space.count_combinations(), 8)

    def test_default_space_is_valid(self) -> None:
        space = SearchSpace.default()
        self.assertTrue(space.is_valid())
        self.assertEqual(
            space.count_combinations(),
            len([100, 150, 200, 300, 400, 500])
            * len([10, 20, 40, 50, 75])
            * len([2, 3, 4, 5, 6]),
        )

    def test_empty_values_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={"chunk_size": [], "chunk_overlap": [10], "top_k": [3]},
        )
        self.assertFalse(space.is_valid())
        self.assertGreaterEqual(len(space.validate()), 1)

    def test_non_integer_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [100, "not-an-int"],
                "chunk_overlap": [20],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())

    def test_non_positive_value_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [-100, 200],
                "chunk_overlap": [20],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())

    def test_duplicate_values_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [100, 100],
                "chunk_overlap": [20],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())

    def test_unsorted_values_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [300, 100],
                "chunk_overlap": [20],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())

    def test_overlap_not_less_than_chunk_size_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [100],
                "chunk_overlap": [100],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())
        self.assertTrue(
            any("chunk_overlap" in e and "chunk_size" in e for e in space.validate())
        )

    def test_overlap_greater_than_chunk_size_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={
                "chunk_size": [100],
                "chunk_overlap": [150],
                "top_k": [3],
            },
        )
        self.assertFalse(space.is_valid())

    def test_raise_if_invalid(self) -> None:
        space = SearchSpace(
            name="bad",
            parameter_values={"chunk_size": [100], "chunk_overlap": [100], "top_k": [3]},
        )
        with self.assertRaises(ValueError):
            space.raise_if_invalid()


class TestCombinationCount(unittest.TestCase):
    def test_count_matches_product(self) -> None:
        space = SearchSpace(
            name="t",
            parameter_values={
                "chunk_size": [100, 200, 300, 400, 500],
                "chunk_overlap": [20, 40],
                "top_k": [2, 3, 4],
            },
        )
        self.assertEqual(space.count_combinations(), 5 * 2 * 3)


class TestDeterministicGeneration(unittest.TestCase):
    def test_generation_order_is_stable(self) -> None:
        space = SearchSpace.pilot()
        first = list(space.generate_parameter_combinations())
        second = list(space.generate_parameter_combinations())
        self.assertEqual(first, second)

    def test_generated_combinations_cover_expected_first_last(self) -> None:
        space = SearchSpace.pilot()
        combos = list(space.generate_parameter_combinations())
        self.assertEqual(combos[0], {"chunk_size": 100, "chunk_overlap": 20, "top_k": 3})
        self.assertEqual(
            combos[-1], {"chunk_size": 300, "chunk_overlap": 50, "top_k": 5}
        )

    def test_to_dict_round_trips(self) -> None:
        space = SearchSpace.pilot()
        d = space.to_dict()
        self.assertEqual(d["name"], "pilot")
        self.assertEqual(d["parameter_values"]["chunk_size"], [100, 300])


if __name__ == "__main__":
    unittest.main()