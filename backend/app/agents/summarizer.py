"""Summarizer agent: synthesises the retrieved sources into Markdown.

The provider call returns Markdown followed by a ``===JSON===`` claim-check block;
only the Markdown half is the summary. The JSON half is discarded here on purpose —
the critic re-derives claim checks against this published text (see
:mod:`app.agents.critic`), so keeping the summarizer's self-assessment would mean
scoring a draft nobody reads.

When no provider is configured the fallback text says so in its own body, so a
mechanical listing is never mistaken for a synthesis.
"""

from __future__ import annotations

from app.models.schemas import ResearchPlan, SourceDocument


class SummarizerAgent:
    def invoke(
        self,
        plan: ResearchPlan,
        documents: list[SourceDocument],
        model_name: str | None = None,
    ) -> str:
        if not documents:
            return "No sources were retrieved for this query, so there is nothing to summarise."

        from app.services.llm_provider import summarize_with_provider, parse_json_section

        raw = summarize_with_provider(plan, documents, model_name=model_name)
        markdown, _claim_checks = parse_json_section(raw)
        return markdown.strip() or raw.strip()
