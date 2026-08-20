import math

from .embeddings import embed_text
from .models import RetrievedChunk
from .vector_store import LocalVectorStore


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return numerator / (left_norm * right_norm)


def retrieve(query: str, store: LocalVectorStore, top_k: int = 3) -> list[RetrievedChunk]:
    query_embedding = embed_text(query, store.vocabulary)

    scored_chunks = [
        RetrievedChunk(
            source=record.source,
            chunk_index=record.chunk_index,
            text=record.text,
            score=cosine_similarity(query_embedding, record.embedding),
        )
        for record in store.records
    ]

    scored_chunks.sort(key=lambda item: item.score, reverse=True)
    return scored_chunks[:top_k]
