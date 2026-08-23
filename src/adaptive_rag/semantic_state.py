from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = Path("storage") / "semantic_embeddings.json"


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the sentence embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embedding_dimension() -> int:
    """Detect the embedding dimension from the model rather than hardcoding."""
    model = _get_model()
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def _load_cache() -> dict[str, list[float]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def question_to_embedding(question: str) -> list[float]:
    """Convert a question into a semantic embedding vector.

    Uses a small pretrained sentence embedding model. Does NOT require Gemini
    or an API key. Embeddings are cached locally so repeated computation is
    avoided.
    """
    cache = _load_cache()
    if question in cache:
        return list(cache[question])

    model = _get_model()
    vector = model.encode(question, convert_to_numpy=True)
    embedding = [float(value) for value in np.asarray(vector).flatten()]

    cache[question] = embedding
    _save_cache(cache)

    return embedding


def clear_cache() -> None:
    """Delete the local embedding cache (mainly for tests)."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


def cache_size() -> int:
    """Return the number of cached embeddings."""
    return len(_load_cache())