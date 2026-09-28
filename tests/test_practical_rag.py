"""Phase 7D practical-flow validation (mocks/stubs only -- no Gemini calls).

End-to-end coverage of the 7A-7C application path:

query -> RAGService (300/40/top_k=2) -> context/sources
  -> optional mocked grounded generation -> answer + metadata.
"""
import unittest


class TestPracticalRetrievalOnly(unittest.TestCase):
    def test_flow_a_retrieval_only(self):
        import app.chat as chat
        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?")
        response, generation = handled["response"], handled["generation"]
        self.assertTrue(response.retrieved_documents)
        self.assertTrue(response.context)
        self.assertTrue(response.sources)
        self.assertIsNone(generation)
        display = chat.format_response(response, generation)
        self.assertFalse(display["generation_used"])
        self.assertIsNone(display["answer"])
        self.assertTrue(display["sources"])
        self.assertEqual(display["config"],
                         {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2})


class TestPracticalGenerationEnabled(unittest.TestCase):
    def test_flow_b_mocked_generation(self):
        import app.chat as chat
        seen = {}

        def stub(query, context):
            seen["query"] = query
            seen["context"] = context
            return "mocked practical answer"

        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?",
            generate=True, generate_fn=stub)
        response, generation = handled["response"], handled["generation"]
        self.assertEqual(seen["context"], response.context)
        self.assertEqual(seen["query"], response.query)
        self.assertTrue(generation.generation_used)
        display = chat.format_response(response, generation)
        self.assertEqual(display["answer"], "mocked practical answer")
        self.assertTrue(display["generation_used"])
        self.assertEqual(display["sources"], response.sources)
        self.assertTrue(display["context"])


class TestPracticalEdgeCases(unittest.TestCase):
    def test_flow_c_invalid_query(self):
        import app.chat as chat
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                chat.handle_query_with_generation(bad)

    def test_flow_d_empty_context_skips_generation(self):
        import app.chat as chat
        from src.adaptive_rag.generation import (
            INSUFFICIENT_CONTEXT_MESSAGE, generate_grounded_answer)

        def exploding(_query, _context):
            raise AssertionError("generator must not be called")

        result = generate_grounded_answer("Q?", "   ", generate_fn=exploding)
        self.assertFalse(result.generation_used)
        self.assertEqual(result.answer, INSUFFICIENT_CONTEXT_MESSAGE)

    def test_flow_e_llm_failure_preserves_retrieval(self):
        import app.chat as chat

        def failing(_query, _context):
            raise RuntimeError("simulated outage")

        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?",
            generate=True, generate_fn=failing)
        response, generation = handled["response"], handled["generation"]
        self.assertFalse(generation.generation_used)
        self.assertIn("simulated outage", generation.error)
        self.assertTrue(response.retrieved_documents)
        self.assertTrue(response.sources)
        display = chat.format_response(response, generation)
        self.assertIn("simulated outage", display["generation_error"])
        self.assertTrue(display["sources"])


class TestPracticalGroundingAndConfig(unittest.TestCase):
    def test_flow_f_grounding_inputs(self):
        import app.chat as chat
        from src.adaptive_rag import rag_service
        captured = {}
        real_query_rag = rag_service.query_rag

        def spy(query, top_k=2):
            response = real_query_rag(query, top_k=top_k)
            captured["context"] = response.context
            captured["query"] = response.query
            return response

        def stub(query, context):
            captured["gen_query"] = query
            captured["gen_context"] = context
            return "grounded"

        original = chat.query_rag
        chat.query_rag = spy
        try:
            handled = chat.handle_query_with_generation(
                "What is the annual leave policy?",
                generate=True, generate_fn=stub)
        finally:
            chat.query_rag = original
        self.assertEqual(captured["gen_context"], captured["context"])
        self.assertEqual(captured["gen_query"], captured["query"])
        self.assertTrue(handled["generation"].generation_used)

    def test_flow_g_validated_configuration(self):
        from src.adaptive_rag import rag_service
        self.assertEqual(rag_service.get_validated_config(),
                         {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2})
        response = rag_service.query_rag("What is the leave policy?")
        self.assertEqual(response.retrieval_metadata["chunk_size"], 300)
        self.assertEqual(response.retrieval_metadata["chunk_overlap"], 40)
        self.assertEqual(response.retrieval_metadata["top_k_requested"], 2)
        self.assertEqual(len(response.retrieved_documents), 2)


if __name__ == "__main__":
    unittest.main()
