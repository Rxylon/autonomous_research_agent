from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import Any
import os

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover - import guard for lightweight environments
        return None

    model_name = settings.embedding_model or "BAAI/bge-small-en-v1.5"
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = text.lower().split()
    for token in tokens:
        digest = sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        value = int.from_bytes(digest[1:5], "little") / 2**32
        vector[index] += value

    magnitude = sum(component * component for component in vector) ** 0.5 or 1.0
    return [component / magnitude for component in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    # If an OpenAI key is configured prefer provider embeddings for quality
    try:
        if settings.openai_api_key:
            try:
                import openai

                openai.api_key = settings.openai_api_key
                model_name = os.getenv("EMBEDDING_MODEL") or settings.embedding_model or "text-embedding-3-small"
                resp = openai.Embedding.create(model=model_name, input=texts)
                embeddings = [item["embedding"] for item in resp["data"]]
                return embeddings
            except Exception:
                pass
    except Exception:
        # ignore and fall back
        pass

    model = get_embedding_model()
    if model is None:
        return [_hash_embedding(text) for text in texts]

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
    except Exception:
        return [_hash_embedding(text) for text in texts]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
