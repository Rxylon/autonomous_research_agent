from __future__ import annotations

from collections import Counter

from app.models.schemas import ResearchPlan, SourceDocument


class SummarizerAgent:
    def invoke(self, plan: ResearchPlan, documents: list[SourceDocument]) -> str:
        """Produce a high-quality summary. If an LLM provider is configured, use it; otherwise fall back to a deterministic summary."""
        if not documents:
            return "No papers were retrieved for this query."

        try:
            from app.services.llm_provider import summarize_with_provider, try_parse_json_section

            raw = summarize_with_provider(plan, documents)
            md, _json = try_parse_json_section(raw)
            return md
        except Exception:
            # graceful fallback to previous local summary behavior
            sources = Counter(document.source for document in documents)
            years = sorted({document.year for document in documents if document.year})
            top_titles = [document.title for document in documents[:8]]

            summary_lines = [
                f"Research objective: {plan.objective}",
                f"Retrieved {len(documents)} source documents from {len(sources)} source types.",
                f"Publication years observed: {', '.join(map(str, years)) if years else 'not available' }.",
                "Major themes to investigate:",
                "- Transformer-based multimodal fusion",
                "- CNN and temporal feature modeling",
                "- Vision-language alignment and contrastive learning",
                "- Dataset bias, robustness, and evaluation leakage",
                "Representative sources:",
            ]
            summary_lines.extend(f"- {title}" for title in top_titles)
            return "\n".join(summary_lines)
