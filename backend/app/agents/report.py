from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.models.schemas import ClaimCheck, ResearchPlan, ReportPayload, SourceDocument


class ReportAgent:
    def invoke(
        self,
        query: str,
        plan: ResearchPlan,
        summary: str,
        documents: list[SourceDocument],
        critic_score: float,
        claim_checks: list[ClaimCheck],
    ) -> ReportPayload:
        reports_directory = Path(settings.reports_directory)
        reports_directory.mkdir(parents=True, exist_ok=True)
        slug = "-".join(query.lower().split()[:6]) or "research-report"
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        pdf_path = reports_directory / f"{slug}-{timestamp}.pdf"

        markdown_sections = [
            f"# Research Report\n\nTopic: {query}\n\n",
            "## Plan\n",
            "\n".join(f"- {step}" for step in plan.steps),
            "\n\n## Summary\n",
            summary,
            f"\n\n## Critic Score\n\n{critic_score:.2f}",
            "\n\n## Evidence\n",
            "\n".join(
                f"- {document.title} ({document.source})" + (f" - {document.url}" if document.url else "")
                for document in documents
            ),
            "\n\n## Claim Checks\n",
            "\n".join(
                f"- {'SUPPORTED' if check.supported else 'UNSUPPORTED'}: {check.claim}" for check in claim_checks
            ),
        ]
        markdown = "".join(markdown_sections)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter
            y = height - 48
            for line in markdown.splitlines():
                pdf.drawString(40, y, line[:110])
                y -= 14
                if y < 48:
                    pdf.showPage()
                    y = height - 48
            pdf.save()
        except Exception:
            pdf_path = None

        return ReportPayload(
            markdown=markdown,
            json_summary={
                "query": query,
                "plan": plan.model_dump(),
                "critic_score": critic_score,
                "documents": [document.model_dump() for document in documents],
                "claim_checks": [check.model_dump() for check in claim_checks],
            },
            pdf_path=str(pdf_path) if pdf_path else None,
        )
