"""Phase 7A RAG service tests (local retrieval-only, no external API)."""
import unittest


class TestRAGService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.adaptive_rag import rag_service
        cls.service = rag_service

    def test_successful_query_retrieval(self):
        response = self.service.query_rag("What is the annual leave policy?")
        self.assertEqual(response.query, "What is the annual leave policy?")
        self.assertTrue(response.retrieved_documents)
        self.assertTrue(response.context)
        self.assertTrue(response.sources)
        self.assertFalse(response.retrieval_metadata["generation_used"])

    def test_top_k_two_results(self):
        response = self.service.query_rag(
            "What is the domestic hotel reimbursement limit?")
        self.assertEqual(response.retrieval_metadata["top_k_requested"], 2)
        self.assertEqual(len(response.retrieved_documents), 2)
        self.assertLessEqual(len(response.sources), 2)

    def test_context_construction(self):
        response = self.service.query_rag("What is the password policy?")
        for doc in response.retrieved_documents:
            self.assertIn(doc["text"], response.context)
            self.assertIn(doc["source"], response.context)

    def test_sources_and_document_ids(self):
        response = self.service.query_rag("Who approves production deployments?")
        for doc in response.retrieved_documents:
            self.assertIn("source", doc)
            self.assertIn("chunk_index", doc)
            self.assertIn("document_id", doc)
            self.assertTrue(doc["document_id"].startswith(doc["source"]))
            self.assertIn("score", doc)
        for source in response.sources:
            self.assertIsInstance(source, str)
            self.assertTrue(source)

    def test_empty_query_rejected(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                self.service.query_rag(bad)

    def test_invalid_top_k_rejected(self):
        for bad in (0, -1, "2", 2.0, None):
            with self.assertRaises(ValueError):
                self.service.query_rag("What is the leave policy?", top_k=bad)

    def test_retrieval_metadata(self):
        response = self.service.query_rag("What is the leave policy?")
        metadata = response.retrieval_metadata
        self.assertEqual(metadata["chunk_size"], 300)
        self.assertEqual(metadata["chunk_overlap"], 40)
        self.assertEqual(metadata["num_returned"],
                         len(response.retrieved_documents))
        self.assertGreater(metadata["num_chunks"], 0)
        self.assertGreater(metadata["num_documents"], 0)
        self.assertGreaterEqual(metadata["retrieval_latency_ms"], 0.0)

    def test_validated_config(self):
        self.assertEqual(self.service.get_validated_config(),
                         {"chunk_size": 300, "chunk_overlap": 40, "top_k": 2})


if __name__ == "__main__":
    unittest.main()
