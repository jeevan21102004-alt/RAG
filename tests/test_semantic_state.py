import unittest

import numpy as np

from src.adaptive_rag.semantic_rl_environment import SemanticRetrievalDecisionEnvironment
from src.adaptive_rag.semantic_state import (
    clear_cache,
    embedding_dimension,
    question_to_embedding,
)


class TestSemanticState(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The first call downloads the pretrained model (all-MiniLM-L6-v2).
        # This requires internet access ONCE to download the model.
        clear_cache()

    def test_same_question_same_embedding(self) -> None:
        question = "What is supervised learning?"
        emb1 = question_to_embedding(question)
        emb2 = question_to_embedding(question)
        self.assertEqual(emb1, emb2)

    def test_embedding_dimension_is_consistent(self) -> None:
        dim = embedding_dimension()
        emb = question_to_embedding("What is machine learning?")
        self.assertEqual(len(emb), dim)

    def test_different_questions_produce_different_vectors(self) -> None:
        emb1 = question_to_embedding("What is machine learning?")
        emb2 = question_to_embedding("What is 2 + 2?")
        self.assertNotEqual(emb1, emb2)

    def test_output_is_numerical(self) -> None:
        emb = question_to_embedding("What is machine learning?")
        arr = np.asarray(emb)
        self.assertTrue(np.issubdtype(arr.dtype, np.floating))

    def test_semantic_environment_returns_correct_state_shape(self) -> None:
        env = SemanticRetrievalDecisionEnvironment(expected_action="SEARCH")
        state = env.reset("What is machine learning?")
        self.assertEqual(len(state), embedding_dimension())


if __name__ == "__main__":
    unittest.main()