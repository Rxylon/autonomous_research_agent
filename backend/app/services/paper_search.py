"""Academic paper discovery across arXiv, Semantic Scholar, and Crossref.

All three backends are plain HTTP with ``urllib``, which blocks. The fan-out here
runs them on a thread pool so one search covers every backend in roughly the time
of the slowest one instead of the sum of all of them. Callers on an event loop must
still offload :meth:`PaperSearchService.search` itself to a worker thread —
``app.graph.workflow`` does that for every node.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from app.core.config import settings
from app.models.schemas import SourceDocument

# Semantic Scholar and Crossref both ask for an identifying User-Agent and will
# rate-limit anonymous traffic more aggressively without one.
_USER_AGENT = "autonomous-multi-research-agent/0.1 (+https://github.com/Rxylon/autonomous_research_agent)"


@dataclass
class SearchResult:
    document: SourceDocument
    raw: dict[str, Any]


def _fetch(url: str, timeout: float | None = None) -> bytes:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=timeout or settings.http_timeout_seconds) as response:
        return response.read()


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
            root = ET.fromstring(
                _fetch(f"https://export.arxiv.org/api/query?{query_string}").decode("utf-8", errors="ignore")
            )
        except Exception:
            return []

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

            authors = [
                author.findtext("atom:name", default="", namespaces=namespace) or ""
                for author in entry.findall("atom:author", namespace)
            ]

            results.append(
                SearchResult(
                    document=SourceDocument(
                        title=title,
                        url=url,
                        authors=[author for author in authors if author],
                        year=year,
                        abstract=summary or None,
                        content=summary or title,
                        source="arxiv",
                    ),
                    raw={"pdf_url": url},
                )
            )

        return results

    def search_semantic_scholar(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,url"}
        try:
            payload = _fetch(
                f"https://api.semanticscholar.org/graph/v1/paper/search?{urlencode(params)}"
            ).decode("utf-8", errors="ignore")
            data = json.loads(payload)
        except Exception:
            return []

        results: list[SearchResult] = []
        for paper in data.get("data") or []:
            results.append(
                SearchResult(
                    document=SourceDocument(
                        title=paper.get("title") or "Untitled paper",
                        url=paper.get("url"),
                        authors=[a.get("name", "") for a in paper.get("authors") or [] if a.get("name")],
                        year=paper.get("year"),
                        abstract=paper.get("abstract"),
                        content=paper.get("abstract") or paper.get("title") or "",
                        source="semantic_scholar",
                    ),
                    raw=paper,
                )
            )
        return results

    def search_crossref(self, query: str, limit: int = 5) -> list[SearchResult]:
        params = {"query.title": query, "rows": limit}
        try:
            payload = _fetch(f"https://api.crossref.org/works?{urlencode(params)}").decode("utf-8", errors="ignore")
            data = json.loads(payload)
        except Exception:
            return []

        results: list[SearchResult] = []
        for item in data.get("message", {}).get("items") or []:
            titles = item.get("title") or []
            title = titles[0] if titles else "Untitled"

            authors = []
            for author in (item.get("author") or [])[:5]:
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)

            year = None
            for key in ("published-print", "published-online", "created"):
                parts = (item.get(key) or {}).get("date-parts") or []
                if parts and parts[0]:
                    year = parts[0][0]
                    break

            abstract = item.get("abstract")
            results.append(
                SearchResult(
                    document=SourceDocument(
                        title=title,
                        url=item.get("URL"),
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        content=abstract or title,
                        source="crossref",
                    ),
                    raw=item,
                )
            )
        return results

    # Historical alias.
    search_web = search_crossref

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Query arXiv and Semantic Scholar in parallel, falling back to Crossref."""
        tasks: list[Callable[[], list[SearchResult]]] = [
            lambda: self.search_arxiv(query, max_results=max_results),
            lambda: self.search_semantic_scholar(query, limit=max_results),
        ]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            batches = list(pool.map(lambda task: task(), tasks))

        documents = [result.document for batch in batches for result in batch]
        if not documents:
            documents = [result.document for result in self.search_crossref(query, limit=max_results)]

        return documents[: max_results * 2]

    def search_many(self, queries: list[str], max_results: int = 5) -> list[SourceDocument]:
        """Run several queries in parallel and return their documents de-duplicated.

        De-duplication is by normalised title, since the same paper routinely comes
        back from more than one query and more than one backend.
        """
        if not queries:
            return []

        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
            batches = list(pool.map(lambda query: self.search(query, max_results=max_results), queries))

        documents: list[SourceDocument] = []
        seen: set[str] = set()
        for batch in batches:
            for document in batch:
                key = " ".join((document.title or "").lower().split())
                if key and key in seen:
                    continue
                seen.add(key)
                documents.append(document)
        return documents
