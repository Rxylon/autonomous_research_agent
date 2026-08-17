"""ChromaDB-backed vector store for retrieved papers and uploaded documents.

The collection name embeds the active embedding backend (see
:func:`app.services.embeddings.embedding_backend`). Vectors from different
backends are not comparable — and OpenAI's are not even the same width — so
mixing them in one collection would either corrupt ranking or make Chroma raise
on a dimension mismatch. Keying the name means switching backends starts a clean
collection instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.models.schemas import SourceDocument
from app.services.embeddings import embed_query, embed_texts, embedding_backend

# Document fields copied into chunk metadata. `content` is deliberately excluded:
# it is already stored as the chunk document, and duplicating it per chunk bloats
# the SQLite metadata table for no benefit.
_METADATA_FIELDS = ("title", "url", "authors", "year", "source")


class VectorStoreService:
    def __init__(self) -> None:
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def collection_name(self) -> str:
        return f"{settings.chroma_collection_name}__{embedding_backend().replace('-', '_')}"

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        return self._client

    def _collection(self):
        return self._get_client().get_or_create_collection(name=self.collection_name)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata values must be str/int/float/bool — flatten or drop the rest."""
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if not value:
                    continue
                sanitized[key] = ", ".join(str(item) for item in value)
                continue
            if isinstance(value, dict):
                continue
            sanitized[key] = value
        return sanitized

    def _chunk_text(self, text: str) -> list[str]:
        text = text or ""
        if not text.strip():
            return []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)
        except Exception:
            chunks = [text[index : index + 1000] for index in range(0, len(text), 800)]
        return [chunk for chunk in chunks if chunk.strip()]

    def count(self) -> int:
        try:
            return self._collection().count()
        except Exception:
            return 0

    def add_documents(self, documents: list[SourceDocument]) -> int:
        """Index each document's content as overlapping chunks. Returns chunks written."""
        if not documents:
            return 0

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for document in documents:
            payload = document.model_dump()
            base_metadata = {field: payload.get(field) for field in _METADATA_FIELDS}
            for chunk_index, chunk in enumerate(self._chunk_text(document.content)):
                texts.append(chunk)
                ids.append(str(uuid4()))
                metadatas.append(self._sanitize_metadata({**base_metadata, "chunk_index": chunk_index}))

        if not texts:
            return 0

        self._collection().add(ids=ids, documents=texts, embeddings=embed_texts(texts), metadatas=metadatas)
        return len(texts)

    def add_text(self, text: str, document: SourceDocument) -> int:
        """Index free text (an upload) under ``document``'s metadata."""
        return self.add_documents([document.model_copy(update={"content": text})])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the ``top_k`` nearest chunks as ``{content, metadata, distance}`` dicts."""
        try:
            collection = self._collection()
            if collection.count() == 0:
                return []
            results = collection.query(
                query_embeddings=[embed_query(query)],
                n_results=min(top_k, collection.count()),
            )
        except Exception:
            return []

        contents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, content in enumerate(contents):
            hits.append(
                {
                    "content": content,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return hits

    def search_documents(self, query: str, top_k: int = 5) -> list[SourceDocument]:
        """Vector search projected back onto :class:`SourceDocument`.

        Chunks belonging to the same title are merged, so a long uploaded PDF comes
        back as one document with its most relevant passages rather than N fragments.
        """
        merged: dict[str, SourceDocument] = {}
        for hit in self.search(query, top_k=top_k):
            metadata = hit.get("metadata") or {}
            title = str(metadata.get("title") or "Untitled document")
            key = " ".join(title.lower().split())
            content = hit.get("content") or ""

            existing = merged.get(key)
            if existing is not None:
                merged[key] = existing.model_copy(
                    update={"content": f"{existing.content}\n\n{content}".strip()}
                )
                continue

            authors = metadata.get("authors")
            year = metadata.get("year")
            merged[key] = SourceDocument(
                title=title,
                url=metadata.get("url") or None,
                authors=[a.strip() for a in str(authors).split(",") if a.strip()] if authors else [],
                year=int(year) if isinstance(year, (int, float)) or (isinstance(year, str) and year.isdigit()) else None,
                abstract=None,
                content=content,
                source=str(metadata.get("source") or "vector_store"),
            )
        return list(merged.values())


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()
