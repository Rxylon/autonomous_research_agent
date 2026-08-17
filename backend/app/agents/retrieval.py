"""Retrieval agent: fresh web search plus a read back out of the vector store.

The flow is index-then-query, which is what makes this retrieval-augmented rather
than just a search wrapper:

1. Search arXiv / Semantic Scholar (Crossref as fallback) for the plan's queries.
2. Index every result into Chroma.
3. Query Chroma with the plan objective. This surfaces two things the raw search
   cannot: passages from documents a user uploaded via ``POST /upload``, and
   passages from papers indexed by earlier runs.
4. Merge, capping the total so downstream prompts stay inside token limits.

Step 3 is why an uploaded PDF actually influences a later answer.
"""

from __future__ import annotations

from pathlib import Path

from app.models.schemas import ResearchPlan, RetrievalResult, SourceDocument
from app.services.paper_search import PaperSearchService
from app.services.vector_store import get_vector_store

# Hard cap on documents handed to the summarizer. Prompts include an ~800 char
# snippet each, so this bounds prompt size regardless of how much search returns.
MAX_DOCUMENTS = 12


def extract_pdf_text(pdf_path: Path | str) -> str:
    """Extract text from a PDF on disk using PyMuPDF.

    Raises if PyMuPDF is unavailable or the file is not a readable PDF — callers
    decide whether to degrade, since silently returning "" looks like an empty PDF.
    """
    import fitz  # imported lazily: PyMuPDF is a heavy native dependency

    document = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


def extract_pdf_text_from_bytes(payload: bytes) -> str:
    """Extract text from PDF bytes (an upload) without touching the filesystem."""
    import fitz

    document = fitz.open(stream=payload, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


class RetrievalAgent:
    def __init__(self) -> None:
        self.paper_search = PaperSearchService()
        self.vector_store = get_vector_store()

    def invoke(self, plan: ResearchPlan, top_k: int = 5) -> RetrievalResult:
        queries = plan.search_queries[:3] or [plan.objective]
        searched = self.paper_search.search_many(queries, max_results=top_k)

        chunks_indexed = 0
        try:
            chunks_indexed = self.vector_store.add_documents(searched)
        except Exception:
            # A vector-store failure must not sink the run; the search results
            # alone are still a usable source set.
            chunks_indexed = 0

        searched_titles = {" ".join((document.title or "").lower().split()) for document in searched}

        recalled: list[SourceDocument] = []
        try:
            for document in self.vector_store.search_documents(plan.objective, top_k=top_k * 2):
                key = " ".join((document.title or "").lower().split())
                if key and key not in searched_titles:
                    recalled.append(document)
        except Exception:
            recalled = []

        documents = (searched + recalled)[:MAX_DOCUMENTS]
        return RetrievalResult(
            documents=documents,
            searched_count=len(searched),
            recalled_count=len(recalled),
            chunks_indexed=chunks_indexed,
            vector_store_size=self.vector_store.count(),
        )
