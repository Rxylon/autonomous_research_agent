"""Embedding service with a three-tier strategy.

1. OpenAI embeddings — only when ``OPENAI_EMBEDDINGS=1`` is set *and* a key exists.
2. A local Sentence-Transformers model (the default; downloads ~130 MB on first use).
3. A deterministic hashed bag-of-words vector, so the vector store still works in a
   minimal environment with no model download.

Tier 3 is a real fallback, not a stand-in for semantic search: hashed vectors only
match on shared tokens. :func:`embedding_backend` reports which tier is live so the
API can surface it rather than letting a caller assume tier 1 or 2.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import Any

from app.core.config import settings

EMBEDDING_DIMENSIONS = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the local Sentence-Transformers model, or None if unavailable/disabled."""
    if settings.embedding_backend_preference in {"hash", "openai"}:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover - import guard for lightweight environments
        return None

    try:
        return SentenceTransformer(settings.embedding_model)
    except Exception:
        # Missing model download, no disk space, or not enough memory to load torch.
        return None


@lru_cache(maxsize=1)
def _openai_embeddings_enabled() -> bool:
    return settings.embedding_backend_preference == "openai" and bool(settings.openai_api_key)


@lru_cache(maxsize=1)
def embedding_backend() -> str:
    """The embedding tier actually in use: ``openai``, ``sentence-transformers``, or ``hash``.

    Cached because the vector store derives its collection name from this, and that
    name must not change mid-process — a flip would split writes and reads across
    two collections.
    """
    if _openai_embeddings_enabled():
        return "openai"
    return "sentence-transformers" if get_embedding_model() is not None else "hash"


def _hash_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "little") % dimensions
        vector[index] += int.from_bytes(digest[2:6], "little") / 2**32

    magnitude = sum(component * component for component in vector) ** 0.5 or 1.0
    return [component / magnitude for component in vector]


def _openai_embeddings(texts: list[str]) -> list[list[float]] | None:
    """Embed via the OpenAI v1+ SDK. Returns None on any failure so callers fall through."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(model=settings.openai_embedding_model, input=texts)
        return [item.embedding for item in response.data]
    except Exception:
        return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if _openai_embeddings_enabled():
        embeddings = _openai_embeddings(texts)
        if embeddings is not None:
            return embeddings

    model = get_embedding_model()
    if model is not None:
        try:
            return [vector.tolist() for vector in model.encode(texts, normalize_embeddings=True)]
        except Exception:
            pass

    return [_hash_embedding(text) for text in texts]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
