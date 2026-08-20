from .models import Chunk, Document


def split_text(text: str, chunk_size_words: int = 60, overlap_words: int = 15) -> list[str]:
    words = text.split()
    if not words:
        return []

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words

    return chunks


def chunk_documents(documents: list[Document], chunk_size_words: int = 60, overlap_words: int = 15) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        document_chunks = split_text(document.text, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
        for chunk_index, chunk_text in enumerate(document_chunks):
            chunks.append(Chunk(source=document.source, chunk_index=chunk_index, text=chunk_text))

    return chunks
