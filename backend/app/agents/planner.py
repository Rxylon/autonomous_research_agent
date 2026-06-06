from __future__ import annotations

from app.models.schemas import ResearchPlan


class PlannerAgent:
    def invoke(self, query: str) -> ResearchPlan:
        search_queries = [query, f'"{query}" recent research', f'{query} survey review']
        return ResearchPlan(
            objective=query,
            steps=[
                "Search recent arXiv and Semantic Scholar papers",
                "Extract methods, datasets, and evaluation trends",
                "Summarize strengths, weaknesses, and open problems",
                "Verify claims against source evidence",
                "Generate a structured report with citations",
            ],
            search_queries=search_queries,
        )
