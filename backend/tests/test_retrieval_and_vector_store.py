"""Tests for the retrieval loop and the vector store's read path.

The central regression: `VectorStoreService.search` had no callers anywhere in the
backend. Documents were embedded and written to Chroma, and nothing ever read them
back — so the "RAG" pipeline had no retrieval-augmentation step, and an uploaded
document could never influence an answer.
"""

from __future__ import annotations

import pytest

from app.agents.retrieval import RetrievalAgent
from app.models.schemas import ResearchPlan, SourceDocument
from app.services.paper_search import PaperSearchService
from app.services.vector_store import VectorStoreService, get_vector_store

pytest.importorskip("chromadb", reason="vector store tests need chromadb installed")


@pytest.fixture
def store() -> VectorStoreService:
    return get_vector_store()


class TestVectorStoreRoundTrip:
    def test_written_documents_can_be_read_back(self, store, documents):
        written = store.add_documents(documents)
        assert written > 0

        hits = store.search("audio-visual fusion for detecting manipulated media", top_k=3)
        assert hits, "search returned nothing for text that was just indexed"
        assert any("fusion" in (hit["content"] or "").lower() for hit in hits)

    def test_search_documents_projects_back_to_source_documents(self, store, documents):
        store.add_documents(documents)
        recalled = store.search_documents("temporal inconsistency forgery signal", top_k=3)
        assert recalled
        assert all(isinstance(document, SourceDocument) for document in recalled)
        assert any(document.source in {"arxiv", "semantic_scholar"} for document in recalled)

    def test_chunks_of_one_document_merge_into_one_result(self, store):
        long_document = SourceDocument(
            title="A Single Long Uploaded Paper",
            content=" ".join(f"section {index} discusses spectral watermarking" for index in range(400)),
            source="upload",
        )
        assert store.add_documents([long_document]) > 1, "expected the text to split into several chunks"

        recalled = store.search_documents("spectral watermarking", top_k=5)
        matching = [d for d in recalled if d.title == "A Single Long Uploaded Paper"]
        assert len(matching) == 1, "chunks of one document should merge, not appear as N documents"

    def test_add_text_indexes_under_supplied_metadata(self, store):
        count = store.add_text(
            "Rotational invariance in graph kernels for molecular property prediction.",
            document=SourceDocument(title="uploaded.pdf", source="upload"),
        )
        assert count >= 1
        recalled = store.search_documents("rotational invariance graph kernels", top_k=3)
        assert any(document.source == "upload" for document in recalled)

    def test_empty_inputs_are_no_ops(self, store):
        assert store.add_documents([]) == 0
        assert store.add_documents([SourceDocument(title="blank", content="   ")]) == 0

    def test_metadata_excludes_duplicated_content(self, store, documents):
        store.add_documents(documents[:1])
        hits = store.search("audio-visual fusion", top_k=1)
        assert "content" not in (hits[0]["metadata"] or {}), (
            "content is already the chunk body; duplicating it into metadata bloats the index"
        )

    def test_collection_name_is_keyed_by_embedding_backend(self, store):
        from app.services.embeddings import embedding_backend

        assert embedding_backend().replace("-", "_") in store.collection_name

    def test_embedding_backend_is_stable_within_a_process(self, store):
        """The collection name derives from this, so a mid-process flip would split
        writes and reads across two collections."""
        from app.services.embeddings import embedding_backend

        assert embedding_backend() == embedding_backend()
        assert store.collection_name == store.collection_name


class TestRetrievalAgent:
    def test_search_results_are_indexed_and_recalled(self, monkeypatch, documents):
        agent = RetrievalAgent()
        monkeypatch.setattr(
            agent.paper_search, "search_many", lambda queries, max_results=5: list(documents)
        )

        result = agent.invoke(
            ResearchPlan(
                objective="multimodal deepfake detection",
                steps=[],
                search_queries=["multimodal deepfake detection"],
            )
        )

        assert result.searched_count == len(documents)
        assert result.chunks_indexed > 0, "retrieved documents were never indexed"
        assert result.vector_store_size > 0
        assert result.documents

    def test_uploaded_document_is_recalled_into_a_later_run(self, monkeypatch, documents):
        """The point of the read path: an upload influences a subsequent query."""
        store = get_vector_store()
        store.add_text(
            "Neuromorphic spike-timing analysis is an unusual forensic cue for synthetic speech.",
            document=SourceDocument(title="my-uploaded-notes.pdf", source="upload"),
        )

        agent = RetrievalAgent()
        monkeypatch.setattr(agent.paper_search, "search_many", lambda queries, max_results=5: list(documents))

        result = agent.invoke(
            ResearchPlan(
                objective="neuromorphic spike-timing analysis forensic cue synthetic speech",
                steps=[],
                search_queries=["synthetic speech forensics"],
            )
        )

        titles = {document.title for document in result.documents}
        assert "my-uploaded-notes.pdf" in titles, "the uploaded document never reached the source set"
        assert result.recalled_count >= 1

    def test_recall_never_duplicates_a_searched_document(self, monkeypatch, documents):
        agent = RetrievalAgent()
        monkeypatch.setattr(agent.paper_search, "search_many", lambda queries, max_results=5: list(documents))
        plan = ResearchPlan(objective=documents[0].title, steps=[], search_queries=["q"])

        agent.invoke(plan)
        result = agent.invoke(plan)  # second run: the docs are already in the store

        titles = [document.title for document in result.documents]
        assert len(titles) == len(set(titles)), f"duplicate documents in source set: {titles}"

    def test_document_count_is_capped(self, monkeypatch):
        from app.agents.retrieval import MAX_DOCUMENTS

        many = [SourceDocument(title=f"Paper {i}", content=f"content {i}", source="arxiv") for i in range(50)]
        agent = RetrievalAgent()
        monkeypatch.setattr(agent.paper_search, "search_many", lambda queries, max_results=5: many)

        result = agent.invoke(ResearchPlan(objective="anything", steps=[], search_queries=["q"]))
        assert len(result.documents) <= MAX_DOCUMENTS

    def test_vector_store_failure_does_not_sink_the_run(self, monkeypatch, documents):
        agent = RetrievalAgent()
        monkeypatch.setattr(agent.paper_search, "search_many", lambda queries, max_results=5: list(documents))
        monkeypatch.setattr(agent.vector_store, "add_documents", lambda docs: (_ for _ in ()).throw(RuntimeError("chroma down")))
        monkeypatch.setattr(agent.vector_store, "search_documents", lambda q, top_k=5: (_ for _ in ()).throw(RuntimeError("chroma down")))

        result = agent.invoke(ResearchPlan(objective="anything", steps=[], search_queries=["q"]))
        assert result.documents == documents
        assert result.chunks_indexed == 0

    def test_falls_back_to_objective_when_plan_has_no_queries(self, monkeypatch, documents):
        seen: list[list[str]] = []
        agent = RetrievalAgent()
        monkeypatch.setattr(
            agent.paper_search,
            "search_many",
            lambda queries, max_results=5: seen.append(queries) or list(documents),
        )
        agent.invoke(ResearchPlan(objective="the objective", steps=[], search_queries=[]))
        assert seen == [["the objective"]]


class TestPaperSearchDeduplication:
    def test_search_many_deduplicates_by_normalised_title(self, monkeypatch):
        service = PaperSearchService()
        duplicate = SourceDocument(title="Same  Paper   Title", content="a", source="arxiv")
        variant = SourceDocument(title="same paper title", content="b", source="semantic_scholar")
        unique = SourceDocument(title="Another Paper", content="c", source="arxiv")

        monkeypatch.setattr(
            service,
            "search",
            lambda query, max_results=5: [duplicate, variant, unique],
        )
        documents = service.search_many(["q1", "q2"], max_results=5)
        assert [document.title for document in documents] == ["Same  Paper   Title", "Another Paper"]

    def test_search_many_with_no_queries(self):
        assert PaperSearchService().search_many([]) == []

    def test_network_failure_yields_no_documents_not_an_exception(self, monkeypatch):
        from app.services import paper_search as module

        monkeypatch.setattr(module, "_fetch", lambda url, timeout=None: (_ for _ in ()).throw(OSError("offline")))
        service = PaperSearchService()
        assert service.search_arxiv("q") == []
        assert service.search_semantic_scholar("q") == []
        assert service.search_crossref("q") == []
        assert service.search("q") == []
