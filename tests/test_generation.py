"""Phase 7C generation tests (mocks/stubs only -- no Gemini calls)."""
import unittest


class TestGroundingPrompt(unittest.TestCase):
    def test_prompt_contains_query_and_context(self):
        from src.adaptive_rag.generation import build_grounding_prompt
        prompt = build_grounding_prompt("What is the leave limit?",
                                        "Leave limit is 24 days.")
        self.assertIn("What is the leave limit?", prompt)
        self.assertIn("Leave limit is 24 days.", prompt)

    def test_prompt_enforces_grounding(self):
        from src.adaptive_rag.generation import (
            INSUFFICIENT_CONTEXT_MESSAGE, build_grounding_prompt)
        prompt = build_grounding_prompt("Q?", "Some context.")
        lowered = prompt.lower()
        self.assertIn("only", lowered)
        self.assertIn("retrieved context", lowered)
        self.assertIn(INSUFFICIENT_CONTEXT_MESSAGE, prompt)

    def test_prompt_rejects_bad_inputs(self):
        from src.adaptive_rag.generation import build_grounding_prompt
        for query, context in (("", "ctx"), ("   ", "ctx"), ("Q?", ""),
                               ("Q?", "   "), (None, "ctx"), ("Q?", None)):
            with self.assertRaises(ValueError):
                build_grounding_prompt(query, context)


class TestGenerateGroundedAnswer(unittest.TestCase):
    def test_query_context_passed_correctly(self):
        from src.adaptive_rag.generation import generate_grounded_answer
        seen = {}

        def stub(query, context):
            seen["query"] = query
            seen["context"] = context
            return "stub answer"

        result = generate_grounded_answer("  My query?  ", "  My context.  ",
                                          generate_fn=stub)
        self.assertEqual(seen, {"query": "My query?",
                                "context": "My context."})
        self.assertEqual(result.answer, "stub answer")
        self.assertTrue(result.generation_used)
        self.assertIsNone(result.error)

    def test_empty_context_skips_generation(self):
        from src.adaptive_rag.generation import (
            INSUFFICIENT_CONTEXT_MESSAGE, generate_grounded_answer)

        def exploding(_query, _context):
            raise AssertionError("generator must not be called")

        for empty in ("", "   "):
            result = generate_grounded_answer("Q?", empty,
                                              generate_fn=exploding)
            self.assertEqual(result.answer, INSUFFICIENT_CONTEXT_MESSAGE)
            self.assertFalse(result.generation_used)
            self.assertIsNotNone(result.error)

    def test_successful_mocked_generation(self):
        from src.adaptive_rag.generation import generate_grounded_answer
        result = generate_grounded_answer(
            "Q?", "Ctx.", generate_fn=lambda q, c: "  grounded  ")
        self.assertEqual(result.answer, "grounded")
        self.assertTrue(result.generation_used)
        self.assertIsNone(result.error)
        self.assertIn("prompt_chars", result.metadata)

    def test_graceful_failure(self):
        from src.adaptive_rag.generation import generate_grounded_answer

        def failing(_query, _context):
            raise RuntimeError("API key missing")

        result = generate_grounded_answer("Q?", "Ctx.", generate_fn=failing)
        self.assertIsNone(result.answer)
        self.assertFalse(result.generation_used)
        self.assertIn("API key missing", result.error)

    def test_empty_answer_treated_as_failure(self):
        from src.adaptive_rag.generation import generate_grounded_answer
        result = generate_grounded_answer(
            "Q?", "Ctx.", generate_fn=lambda q, c: "   ")
        self.assertIsNone(result.answer)
        self.assertFalse(result.generation_used)

    def test_no_real_llm_import_at_module_load(self):
        import sys
        for module in ("src.adaptive_rag.llm", "adaptive_rag.llm"):
            sys.modules.pop(module, None)
        import src.adaptive_rag.generation as generation
        self.assertTrue(hasattr(generation, "generate_grounded_answer"))


if __name__ == "__main__":
    unittest.main()
