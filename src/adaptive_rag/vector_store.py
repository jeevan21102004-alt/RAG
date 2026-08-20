import json
from pathlib import Path

from .embeddings import build_vocabulary, embed_text
from .models import Chunk, VectorRecord


class LocalVectorStore:
    def __init__(self, vocabulary: list[str], records: list[VectorRecord]) -> None:
        self.vocabulary = vocabulary
        self.records = records

    @classmethod
    def build_from_chunks(cls, chunks: list[Chunk]) -> "LocalVectorStore":
        vocabulary = build_vocabulary([chunk.text for chunk in chunks])
        records = [
            VectorRecord(
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=embed_text(chunk.text, vocabulary),
            )
            for chunk in chunks
        ]
        return cls(vocabulary=vocabulary, records=records)

    @classmethod
    def load(cls, path: Path) -> "LocalVectorStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [VectorRecord(**record) for record in data["records"]]
        return cls(vocabulary=data["vocabulary"], records=records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocabulary": self.vocabulary,
            "records": [
                {
                    "source": record.source,
                    "chunk_index": record.chunk_index,
                    "text": record.text,
                    "embedding": record.embedding,
                }
                for record in self.records
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
