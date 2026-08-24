"""Tests for ExperimentConfig."""

import json
import unittest

from src.adaptive_rag.experiment_config import ExperimentConfig


class TestExperimentConfigCreation(unittest.TestCase):
    def test_config_creation(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test Experiment",
            description="A test experiment",
            parameters={"top_k": 5, "chunk_size": 300},
            tags=["test", "mock"],
        )
        self.assertEqual(config.experiment_id, "exp-001")
        self.assertEqual(config.name, "Test Experiment")
        self.assertEqual(config.description, "A test experiment")
        self.assertEqual(config.parameters, {"top_k": 5, "chunk_size": 300})
        self.assertEqual(config.tags, ["test", "mock"])

    def test_config_defaults(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-002",
            name="Minimal Config",
        )
        self.assertEqual(config.description, "")
        self.assertEqual(config.parameters, {})
        self.assertEqual(config.tags, [])


class TestExperimentConfigSerialization(unittest.TestCase):
    def test_to_dict(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
            tags=["a"],
        )
        d = config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["experiment_id"], "exp-001")
        self.assertEqual(d["name"], "Test")
        self.assertEqual(d["parameters"], {"top_k": 3})
        self.assertEqual(d["tags"], ["a"])

    def test_to_json(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
        )
        json_str = config.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["experiment_id"], "exp-001")

    def test_json_serializable(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3, "strategy": "vector"},
            tags=["test"],
        )
        json.dumps(config.to_dict())


class TestExperimentConfigImmutability(unittest.TestCase):
    def test_frozen_dataclass(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            parameters={"top_k": 3},
        )
        with self.assertRaises(Exception):
            config.experiment_id = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
