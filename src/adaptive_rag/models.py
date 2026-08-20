from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class VectorRecord:
    source: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    chunk_index: int
    text: str
    score: float
