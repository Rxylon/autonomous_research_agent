"""Report agent: renders a run as Markdown, JSON, and PDF.

All three artifacts are written to ``REPORTS_DIRECTORY`` and their paths returned,
so ``GET /reports/{run_id}.{md,json,pdf}`` can serve them. On a host with an
ephemeral filesystem (Render free, Vercel) those files vanish on restart, which is
why the Markdown and JSON bodies are also stored inline in the run history —
those two survive as long as ``history.jsonl`` does, and the download routes fall
back to them. The PDF is the one artifact that cannot be reconstructed from
history and is genuinely lost on restart.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.models.schemas import ClaimCheck, ReportPayload, ResearchPlan, SourceDocument

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_words: int = 6) -> str:
    words = _SLUG_PATTERN.sub("-", text.lower()).strip("-").split("-")
    slug = "-".join(word for word in words if word)[:80].strip("-")
    return "-".join(slug.split("-")[:max_words]) or "research-report"


class ReportAgent:
    def invoke(
        self,
        query: str,
        plan: ResearchPlan,
        summary: str,
        documents: list[SourceDocument],
        critic_score: float,
        claim_checks: list[ClaimCheck],
        critic_method: str = "empty",
    ) -> ReportPayload:
        reports_directory = Path(settings.reports_directory)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        stem = f"{_slugify(query)}-{timestamp}"

        markdown = self._render_markdown(query, plan, summary, documents, critic_score, claim_checks, critic_method)
        json_summary = {
            "query": query,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "plan": plan.model_dump(),
            "summary": summary,
            "critic_score": critic_score,
            "critic_method": critic_method,
            "documents": [document.model_dump() for document in documents],
            "claim_checks": [check.model_dump() for check in claim_checks],
        }

        # Artifact writing is best-effort: a read-only or full filesystem must not
        # fail a run whose content is already complete in memory.
        markdown_path = self._write_text(reports_directory / f"{stem}.md", markdown)
        json_path = self._write_text(reports_directory / f"{stem}.json", json.dumps(json_summary, indent=2, default=str))
        pdf_path = self._write_pdf(reports_directory / f"{stem}.pdf", markdown)

        return ReportPayload(markdown=markdown, json_summary=json_summary, pdf_path=pdf_path)

    def _render_markdown(
        self,
        query: str,
        plan: ResearchPlan,
        summary: str,
        documents: list[SourceDocument],
        critic_score: float,
        claim_checks: list[ClaimCheck],
        critic_method: str,
    ) -> str:
        method_note = {
            "llm": "verified by the configured LLM against the retrieved sources",
            "heuristic": "keyword heuristic only — no model verified these claims",
            "empty": "no claims were extracted, so this score carries no information",
        }.get(critic_method, critic_method)

        lines = [
            "# Research Report",
            "",
            f"**Topic:** {query}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Plan",
            "",
        ]
        lines += [f"{index}. {step}" for index, step in enumerate(plan.steps, start=1)]
        lines += ["", "## Summary", "", summary or "_No summary produced._", "", "## Critic Score", ""]
        lines += [f"**{critic_score:.2f}** ({method_note})", "", "## Sources", ""]

        if documents:
            for index, document in enumerate(documents, start=1):
                citation = f"{index}. **{document.title}** — {document.source}"
                if document.year:
                    citation += f", {document.year}"
                if document.url:
                    citation += f"  \n   <{document.url}>"
                lines.append(citation)
        else:
            lines.append("_No sources retrieved._")

        lines += ["", "## Claim Checks", ""]
        if claim_checks:
            for check in claim_checks:
                lines.append(f"- **{'SUPPORTED' if check.supported else 'UNSUPPORTED'}** — {check.claim}")
                if check.rationale:
                    lines.append(f"  - Rationale: {check.rationale}")
                for evidence in check.evidence:
                    lines.append(f"  - Evidence: {evidence}")
        else:
            lines.append("_No claim checks were produced._")

        return "\n".join(lines) + "\n"

    def _write_text(self, path: Path, content: str) -> str | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def _write_pdf(self, path: Path, markdown: str) -> str | None:
        """Render the Markdown to a paginated PDF with real line wrapping.

        The previous implementation truncated every line at 110 characters, which
        silently dropped the tail of long abstracts and URLs.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfbase.pdfmetrics import stringWidth
            from reportlab.pdfgen import canvas

            path.parent.mkdir(parents=True, exist_ok=True)
            pdf = canvas.Canvas(str(path), pagesize=letter)
            width, height = letter
            left, right, top, bottom = 48, 48, 48, 56
            usable_width = width - left - right
            font, size, leading = "Helvetica", 9.5, 13.0

            pdf.setFont(font, size)
            y = height - top

            for raw_line in markdown.splitlines():
                # Strip Markdown emphasis and heading markers; a flat PDF is more
                # readable than one littered with ** and #.
                line = re.sub(r"\*\*(.+?)\*\*", r"\1", raw_line)
                line = re.sub(r"^#{1,6}\s*", "", line)

                for segment in self._wrap(line, usable_width, font, size, stringWidth) or [""]:
                    if y < bottom:
                        pdf.showPage()
                        pdf.setFont(font, size)
                        y = height - top
                    pdf.drawString(left, y, segment)
                    y -= leading

            pdf.save()
            return str(path)
        except Exception:
            return None

    @staticmethod
    def _wrap(line: str, usable_width: float, font: str, size: float, string_width) -> list[str]:
        if not line.strip():
            return [""]

        indent = line[: len(line) - len(line.lstrip())]
        words, current, out = line.split(), "", []
        for word in words:
            candidate = f"{current} {word}".strip()
            if string_width(indent + candidate, font, size) <= usable_width or not current:
                current = candidate
            else:
                out.append(indent + current)
                current = word
        if current:
            out.append(indent + current)
        return out
