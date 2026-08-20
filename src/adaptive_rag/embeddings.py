from collections import Counter
import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: list[str]) -> list[str]:
    vocabulary: list[str] = []
    seen: set[str] = set()

    for text in texts:
        for token in tokenize(text):
            if token not in seen:
                seen.add(token)
                vocabulary.append(token)

    return vocabulary


def embed_text(text: str, vocabulary: list[str]) -> list[float]:
    token_counts = Counter(tokenize(text))
    vector = [float(token_counts.get(term, 0)) for term in vocabulary]

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector

    return [value / norm for value in vector]
