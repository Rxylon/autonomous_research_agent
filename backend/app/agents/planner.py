"""Planner agent: turns a query into search queries and a fixed step list.

This planner is **rule-based, not LLM-driven**. It strips conversational filler
from the query and expands it into a few search variants; the five steps it returns
are a constant, describing the pipeline the other agents will run.

That is a deliberate trade-off, not an unfinished piece: it costs no tokens, adds
no latency, and never fails — which matters on a free-tier host where the whole
run has to fit inside one request timeout. The consequence is that planning does
not adapt to the query beyond keyword extraction, so a question needing genuinely
different sub-investigations gets the same three search variants as any other.
"""

from __future__ import annotations

import re

from app.models.schemas import ResearchPlan

# Leading conversational phrasing that hurts keyword search relevance.
_FILLER_PREFIX = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:find|explain|summari[sz]e|describe|compare|tell me about|give me|list|what are|what is)\s+"
    r"(?:the\s+|some\s+|any\s+)?(?:recent\s+)?(?:advances?\s+in\s+|research\s+on\s+|work\s+on\s+)?",
    re.IGNORECASE,
)

_STEPS = [
    "Search recent arXiv and Semantic Scholar papers",
    "Index sources and recall related passages from the vector store",
    "Summarize methods, datasets, and evaluation trends",
    "Verify each summary claim against source evidence",
    "Generate a structured report with citations",
]


class PlannerAgent:
    def invoke(self, query: str) -> ResearchPlan:
        topic = _FILLER_PREFIX.sub("", query.strip()).strip(" .?!") or query.strip()

        # De-duplicated but order-preserving: the bare topic should be searched first.
        candidates = [topic, f"{topic} survey review", f"{topic} recent advances benchmark"]
        search_queries: list[str] = []
        for candidate in candidates:
            candidate = " ".join(candidate.split())
            if candidate and candidate not in search_queries:
                search_queries.append(candidate)

        return ResearchPlan(objective=query.strip(), steps=list(_STEPS), search_queries=search_queries)
