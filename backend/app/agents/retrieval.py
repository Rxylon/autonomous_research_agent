from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fitz

from app.models.schemas import ResearchPlan, SourceDocument
from app.services.history_store import HistoryStore
from app.services.paper_search import PaperSearchService
from app.services.vector_store import get_vector_store


class RetrievalAgent:
    def __init__(self) -> None:
        self.paper_search = PaperSearchService()
        self.vector_store = get_vector_store()
        self.history_store = HistoryStore()

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        document = fitz.open(pdf_path)
        try:
            return "\n".join(page.get_text() for page in document)
        finally:
            document.close()

    def invoke(self, plan: ResearchPlan, top_k: int = 5) -> list[SourceDocument]:
        documents = []
        for query in plan.search_queries[:3]:
            documents.extend(self.paper_search.search(query, max_results=top_k))

        self.vector_store.add_documents(documents)
        return documents
