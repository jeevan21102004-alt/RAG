"""Tests for the RL optimization environment (Phase 4)."""

import unittest

from src.adaptive_rag.evaluation_schema import EvaluationCase, SystemResponse
from src.adaptive_rag.experiment_config import ExperimentConfig
from src.adaptive_rag.rl_optimization_environment import (
    AlreadyEvaluatedError,
    EpisodeDoneError,
    InvalidActionError,
    RetrievalOptimizationEnvironment,
)
from src.adaptive_rag.search_space import SearchSpace


def _cases(n=2):
    return [
        EvaluationCase(
            case_id=f"ent-00{i}",
            question=f"Q{i}",
            category="retrieval_required",
            expected_answer=f"Answer {i}",
            expected_action="SEARCH",
            relevant_documents=[f"doc{i}.md"],
        )
        for i in range(1, n + 1)
    ]


class DeterministicMockAdapter:
    """Deterministic local adapter; no API calls."""

    def run(self, case: EvaluationCase, config: ExperimentConfig) -> SystemResponse:
        top_k = config.parameters.get("top_k", 2)
        retrieved = list(case.relevant_documents)[:top_k]
        return SystemResponse(
            answer=None,
            action="SEARCH",
            retrieved_documents=retrieved,
            retrieved_context=" ".join(retrieved) or None,
            latency_ms=50.0,
            retrieval_attempts=1,
            metadata={"adapter": "opt_mock"},
        )


def _make_env(budget=4, space=None, cases=None, adapter=None):
    return RetrievalOptimizationEnvironment(
        search_space=space if space is not None else SearchSpace.adaptive_pilot(),
        cases=cases if cases is not None else _cases(),
        adapter=adapter if adapter is not None else DeterministicMockAdapter(),
        budget=budget,
    )


class TestReset(unittest.TestCase):
    def test_reset_returns_state(self) -> None:
        env = _make_env()
        state = env.reset()
        self.assertEqual(len(state.to_vector()), 15)
        self.assertEqual(env.evaluations_used, 0)
        self.assertFalse(env.done)
        self.assertIsNone(env.current_best_objective)

    def test_reset_clears_history(self) -> None:
        env = _make_env()
        env.step(0)
        self.assertEqual(env.evaluations_used, 1)
        env.reset()
        self.assertEqual(env.evaluations_used, 0)
        self.assertEqual(len(env._order), 0)


class TestStep(unittest.TestCase):
    def test_valid_action(self) -> None:
        env = _make_env(budget=4)
        state, reward, done, meta = env.step(0)
        self.assertEqual(env.evaluations_used, 1)
        self.assertIsInstance(reward, float)
        self.assertIsNotNone(meta["objective"])
        self.assertFalse(done)

    def test_invalid_action_out_of_range(self) -> None:
        env = _make_env()
        with self.assertRaises(InvalidActionError):
            env.step(175)
        with self.assertRaises(InvalidActionError):
            env.step(-1)

    def test_duplicate_action_rejected(self) -> None:
        env = _make_env(budget=4)
        env.step(0)
        with self.assertRaises(AlreadyEvaluatedError):
            env.step(0)

    def test_budget_termination(self) -> None:
        env = _make_env(budget=2)
        env.step(0)
        self.assertFalse(env.done)
        env.step(1)
        self.assertTrue(env.done)

    def test_step_after_done_raises(self) -> None:
        env = _make_env(budget=1)
        env.step(0)
        self.assertTrue(env.done)
        with self.assertRaises(EpisodeDoneError):
            env.step(1)

    def test_reward_tracked(self) -> None:
        env = _make_env(budget=2)
        _, r1, _, _ = env.step(0)
        self.assertEqual(env.cumulative_reward, r1)
        _, r2, _, _ = env.step(1)
        self.assertAlmostEqual(env.cumulative_reward, r1 + r2)

    def test_next_state_updates(self) -> None:
        env = _make_env(budget=2)
        s0 = env.reset()
        s1, _, _, _ = env.step(0)
        self.assertNotEqual(s0.best_objective, s1.best_objective)
        self.assertGreater(s1.budget_consumed_fraction, 0.0)

    def test_valid_mask_after_eval(self) -> None:
        env = _make_env(budget=2)
        mask = env.valid_action_mask()
        self.assertTrue(mask[0])
        env.step(0)
        mask = env.valid_action_mask()
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])


if __name__ == "__main__":
    unittest.main()