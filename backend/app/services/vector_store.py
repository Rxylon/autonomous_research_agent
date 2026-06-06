from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.models.schemas import SourceDocument
from app.services.embeddings import embed_query, embed_texts


class VectorStoreService:
    def __init__(self) -> None:
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        return self._client

    def _collection(self):
        return self._get_client().get_or_create_collection(name=settings.chroma_collection_name)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                sanitized[key] = ", ".join(str(item) for item in value)
                continue
            if isinstance(value, dict):
                continue
            sanitized[key] = value
        return sanitized

    def _chunk_text(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_text(text)
        except Exception:
            return [text[i : i + 1000] for i in range(0, max(len(text), 1), 800)]

    def add_documents(self, documents: list[SourceDocument]) -> int:
        if not documents:
            return 0

        collection = self._collection()
        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for document in documents:
            chunks = self._chunk_text(document.content)
            for chunk_index, chunk in enumerate(chunks):
                texts.append(chunk)
                ids.append(str(uuid4()))
                metadatas.append(self._sanitize_metadata({**document.model_dump(), "chunk_index": chunk_index}))

        embeddings = embed_texts(texts)
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return len(texts)

    def add_text(self, text: str, document: SourceDocument) -> int:
        chunked_text = self._chunk_text(text)
        chunk_documents = [
            SourceDocument(
                title=document.title,
                url=document.url,
                authors=document.authors,
                year=document.year,
                abstract=document.abstract,
                content=chunk,
                source=document.source,
            )
            for chunk in chunked_text
        ]
        return self.add_documents(chunk_documents)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        collection = self._collection()
        if collection.count() == 0:
            return []
        results = collection.query(query_embeddings=[embed_query(query)], n_results=top_k)
        documents = []
        for index, content in enumerate(results.get("documents", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][index] if results.get("metadatas") else {}
            distance = results.get("distances", [[]])[0][index] if results.get("distances") else None
            documents.append({"content": content, "metadata": metadata, "distance": distance})
        return documents


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()
