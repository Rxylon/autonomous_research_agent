from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
from typing import Any
from xml.etree import ElementTree as ET
from urllib.parse import urlencode
from urllib.request import urlopen

from app.models.schemas import SourceDocument


@dataclass
class SearchResult:
    document: SourceDocument
    raw: dict[str, Any]


class PaperSearchService:
    def search_arxiv(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            query_string = urlencode(
                {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }
            )
            with urlopen(f"https://export.arxiv.org/api/query?{query_string}", timeout=5.0) as response:
                root = ET.fromstring(response.read().decode("utf-8", errors="ignore"))
            namespace = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            results: list[SearchResult] = []

            for entry in root.findall("atom:entry", namespace):
                title = (entry.findtext("atom:title", default="Untitled paper", namespaces=namespace) or "").strip().replace("\n", " ")
                summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
                url = entry.findtext("atom:id", default=None, namespaces=namespace)
                published_text = entry.findtext("atom:published", default="", namespaces=namespace)
                year = None
                if published_text:
                    try:
                        year = datetime.fromisoformat(published_text.replace("Z", "+00:00")).astimezone(timezone.utc).year
                    except ValueError:
                        year = None

                authors = [author.findtext("atom:name", default="", namespaces=namespace) or "" for author in entry.findall("atom:author", namespace)]
                authors = [author for author in authors if author]

                results.append(
                    SearchResult(
                        document=SourceDocument(
                            title=title,
                            url=url,
                            authors=authors,
                            year=year,
                            abstract=summary or None,
                            content=summary or title,
                            source="arxiv",
                        ),
                        raw={"pdf_url": url},
                    )
                )

            return results
        except Exception:
            return []

    def search_semantic_scholar(self, query: str, limit: int = 5) -> list[SearchResult]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,url"}
        try:
            query_string = urlencode(params)
            with urlopen(f"{url}?{query_string}", timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception:
            return []

        papers = data.get("data", [])
        results: list[SearchResult] = []
        for paper in papers:
            results.append(
                SearchResult(
                    document=SourceDocument(
                        title=paper.get("title", "Untitled paper"),
                        url=paper.get("url"),
                        authors=[author.get("name", "") for author in paper.get("authors", []) if author.get("name")],
                        year=paper.get("year"),
                        abstract=paper.get("abstract"),
                        content=paper.get("abstract") or paper.get("title", ""),
                        source="semantic_scholar",
                    ),
                    raw=paper,
                )
            )
        return results

    def search_web(self, query: str) -> list[SearchResult]:
        # Try CrossRef as a general web/paper discovery fallback
        try:
            url = "https://api.crossref.org/works"
            params = {"query.title": query, "rows": 5}
            query_string = urlencode(params)
            with urlopen(f"{url}?{query_string}", timeout=10.0) as response:
                data = json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception:
            return []

        items = data.get("message", {}).get("items", [])
        results: list[SearchResult] = []
        for it in items:
            title = it.get("title", ["Untitled"])[0] if it.get("title") else "Untitled"
            url = None
            if it.get("URL"):
                url = it.get("URL")
            authors = []
            for a in it.get("author", [])[:5]:
                name = "".join([a.get("given", ""), " ", a.get("family", "")]).strip()
                if name:
                    authors.append(name)
            year = None
            if it.get("published-print") and it.get("published-print").get("date-parts"):
                year = it.get("published-print").get("date-parts")[0][0]
            elif it.get("published-online") and it.get("published-online").get("date-parts"):
                year = it.get("published-online").get("date-parts")[0][0]
            abstract = None
            # Crossref may include some abstract in 'abstract' field but often not
            if it.get("abstract"):
                abstract = it.get("abstract")

            results.append(
                SearchResult(
                    document=SourceDocument(
                        title=title,
                        url=url,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        content=abstract or title,
                        source="crossref",
                    ),
                    raw=it,
                )
            )
        return results

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for result in self.search_arxiv(query, max_results=max_results):
            documents.append(result.document)
        for result in self.search_semantic_scholar(query, limit=max_results):
            documents.append(result.document)
        if not documents:
            # try CrossRef as a broader discovery fallback
            for result in self.search_web(query):
                documents.append(result.document)
        return documents[: max_results * 2]
