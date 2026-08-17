"""Critic agent: grades the published summary against the retrieved sources.

Two properties matter here and both were previously wrong:

* The critic reviews **the summary the user sees**. It calls
  :func:`critique_with_provider` with that exact text rather than asking the model
  for an independent second draft — grading a parallel draft says nothing about
  what was published.
* A model verdict of ``supported: false`` is **final**. Attaching evidence
  snippets does not promote a claim to supported; that inversion pushed every
  score toward 1.0 and made the number meaningless.

:attr:`CriticVerdict.method` records whether an LLM or the keyword heuristic
produced the score, so a caller can tell a verified 0.8 from a guessed one.
"""

from __future__ import annotations

from app.models.schemas import ClaimCheck, CriticVerdict, SourceDocument

# Longest evidence snippet kept per claim, so a whole abstract cannot be pasted in.
_MAX_SNIPPET_CHARS = 300


class CriticAgent:
    def invoke(
        self,
        summary: str,
        documents: list[SourceDocument],
        model_name: str | None = None,
    ) -> CriticVerdict:
        if not summary or not summary.strip():
            return CriticVerdict(score=0.0, claim_checks=[], method="empty")

        from app.services.llm_provider import critique_with_provider, parse_json_section

        raw = critique_with_provider(summary, documents, model_name=model_name)
        _markdown, data = parse_json_section(raw)

        if data.get("error") or "claim_checks" not in data:
            return CriticVerdict(score=0.0, claim_checks=[], method="empty")

        checks = self._build_checks(data.get("claim_checks") or [], documents)
        if not checks:
            return CriticVerdict(score=0.0, claim_checks=[], method="empty")

        # The provider stamps `method: heuristic` when no model produced the verdict.
        method = "heuristic" if data.get("method") == "heuristic" else "llm"

        supported = sum(1 for check in checks if check.supported)
        return CriticVerdict(
            score=round(supported / len(checks), 2),
            claim_checks=checks,
            method=method,
        )

    def _build_checks(self, raw_checks: list, documents: list[SourceDocument]) -> list[ClaimCheck]:
        checks: list[ClaimCheck] = []
        for item in raw_checks:
            if not isinstance(item, dict):
                # A bare string carries no verdict, so it cannot count as supported.
                checks.append(ClaimCheck(claim=str(item), supported=False, rationale="No verdict returned for this claim."))
                continue

            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue

            evidence = self._resolve_evidence(item, documents)
            checks.append(
                ClaimCheck(
                    claim=claim,
                    # The model's verdict stands as given. Evidence is context for a
                    # reader, never a reason to overturn `supported: false`.
                    supported=bool(item.get("supported")),
                    evidence=evidence,
                    rationale=str(item.get("rationale") or ""),
                )
            )
        return checks

    def _resolve_evidence(self, item: dict, documents: list[SourceDocument]) -> list[str]:
        """Turn 1-based source indices and model snippets into readable citations.

        Out-of-range indices are dropped rather than clamped: a model citing source
        [9] of 5 has hallucinated a citation, and clamping would hide that.
        """
        evidence: list[str] = []

        indices = item.get("evidence")
        if isinstance(indices, list):
            for index in indices:
                if not isinstance(index, int) or not 1 <= index <= len(documents):
                    continue
                document = documents[index - 1]
                snippet = (document.abstract or document.content or "")[:200].replace("\n", " ").strip()
                citation = f"[{index}] {document.title}" + (f" — {snippet}" if snippet else "")
                if citation not in evidence:
                    evidence.append(citation)

        snippets = item.get("evidence_snippets")
        if isinstance(snippets, list):
            for snippet in snippets:
                text = str(snippet).strip()[:_MAX_SNIPPET_CHARS]
                if text and text not in evidence:
                    evidence.append(text)

        return evidence
