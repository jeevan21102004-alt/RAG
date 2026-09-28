"""Phase 7B chat interface tests (no Streamlit server, no external API)."""
import unittest


class TestChatModule(unittest.TestCase):
    def test_importable(self):
        import app.chat as chat
        self.assertTrue(hasattr(chat, "handle_query"))
        self.assertTrue(hasattr(chat, "format_response"))
        self.assertTrue(hasattr(chat, "main"))

    def test_handle_query_uses_validated_config(self):
        import app.chat as chat
        response = chat.handle_query("What is the annual leave policy?")
        self.assertEqual(response.query, "What is the annual leave policy?")
        self.assertEqual(len(response.retrieved_documents), 2)
        self.assertEqual(
            response.retrieval_metadata["top_k_requested"], 2)
        self.assertFalse(response.retrieval_metadata["generation_used"])

    def test_format_response_sections(self):
        import app.chat as chat
        response = chat.handle_query("What is the password policy?")
        display = chat.format_response(response)
        self.assertEqual(display["query"], response.query)
        self.assertIn("disabled", display["notice"].lower())
        self.assertEqual(display["sources"], response.sources)
        self.assertEqual(len(display["chunks"]),
                         len(response.retrieved_documents))
        self.assertEqual(display["context"], response.context)
        self.assertIsNone(display["answer"])
        self.assertFalse(display["generation_used"])
        self.assertIn("chunk_size=300", display["metadata"])
        self.assertEqual(display["config"],
                         {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2})

    def test_empty_query_raises(self):
        import app.chat as chat
        with self.assertRaises(ValueError):
            chat.handle_query("   ")

    def test_no_llm_dependency(self):
        import ast
        from pathlib import Path
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "app" / "chat.py")
            .read_text(encoding="utf-8"))
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    called.add(func.id)
                elif isinstance(func, ast.Attribute):
                    called.add(func.attr)
        for banned in ("genai", "openai", "requests", "urllib"):
            self.assertNotIn(banned, imported)
            self.assertNotIn(banned, called)
        self.assertNotIn("generate_text", called)


class TestChatGenerationToggle(unittest.TestCase):
    def test_generation_disabled_makes_no_llm_call(self):
        import app.chat as chat
        calls = []

        def exploding(_query, _context):
            calls.append((_query, _context))
            raise AssertionError("LLM must not be called")

        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?",
            generate=False, generate_fn=exploding)
        self.assertIsNone(handled["generation"])
        self.assertEqual(calls, [])
        self.assertEqual(len(handled["response"].retrieved_documents), 2)
        display = chat.format_response(handled["response"],
                                       handled["generation"])
        self.assertFalse(display["generation_used"])
        self.assertIsNone(display["answer"])

    def test_generation_enabled_invokes_mocked_generator(self):
        import app.chat as chat
        seen = {}

        def stub(query, context):
            seen["query"] = query
            seen["context"] = context
            return "mocked grounded answer"

        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?",
            generate=True, generate_fn=stub)
        response = handled["response"]
        generation = handled["generation"]
        self.assertEqual(seen["context"], response.context)
        self.assertEqual(seen["query"], response.query)
        self.assertTrue(generation.generation_used)
        display = chat.format_response(response, generation)
        self.assertTrue(display["generation_used"])
        self.assertEqual(display["answer"], "mocked grounded answer")
        self.assertEqual(display["sources"], response.sources)

    def test_generation_failure_still_exposes_retrieval(self):
        import app.chat as chat

        def failing(_query, _context):
            raise RuntimeError("no key")

        handled = chat.handle_query_with_generation(
            "What is the annual leave policy?",
            generate=True, generate_fn=failing)
        self.assertFalse(handled["generation"].generation_used)
        display = chat.format_response(handled["response"],
                                       handled["generation"])
        self.assertFalse(display["generation_used"])
        self.assertIn("no key", display["generation_error"])
        self.assertTrue(display["sources"])


if __name__ == "__main__":
    unittest.main()
