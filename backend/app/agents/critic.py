from __future__ import annotations

from app.models.schemas import ClaimCheck, SourceDocument


class CriticAgent:
    def invoke(self, summary: str, documents: list[SourceDocument]) -> tuple[float, list[ClaimCheck]]:
        """Use an LLM provider to perform claim-checking when available, otherwise fall back to a lightweight heuristic."""
        if not summary:
            return 1.0, []

        try:
            from app.services.llm_provider import summarize_with_provider, try_parse_json_section

            # ask the provider to return claim checks in the JSON section
            raw = summarize_with_provider(None, documents)
            _md, data = try_parse_json_section(raw)
            checks: list[ClaimCheck] = []
            claim_checks = data.get("claim_checks") or []
            unsupported = 0
            for item in claim_checks:
                claim = item.get("claim") if isinstance(item, dict) else str(item)
                evidence_idxs = item.get("evidence") if isinstance(item, dict) else []
                evidence_snips = item.get("evidence_snippets") if isinstance(item, dict) else []
                rationale = item.get("rationale") if isinstance(item, dict) else ""

                evidence_list: list[str] = []
                # map integer indices to source snippets/titles
                if isinstance(evidence_idxs, list):
                    for idx in evidence_idxs:
                        try:
                            if isinstance(idx, int) and 1 <= idx <= len(documents):
                                doc = documents[idx - 1]
                                title = getattr(doc, "title", "")
                                snippet = getattr(doc, "abstract", None) or (getattr(doc, "content", "")[:200] if getattr(doc, "content", "") else "")
                                evidence_list.append(f"[{idx}] {title} - {snippet}")
                        except Exception:
                            continue
                # include any snippets returned by the provider
                if isinstance(evidence_snips, list):
                    for s in evidence_snips:
                        try:
                            if s and s not in evidence_list:
                                evidence_list.append(s)
                        except Exception:
                            pass

                supported = bool(item.get("supported")) if isinstance(item, dict) else False
                # treat presence of evidence as support
                if evidence_list:
                    supported = True
                if not supported:
                    unsupported += 1
                checks.append(
                    ClaimCheck(claim=claim, supported=supported, evidence=evidence_list or [], rationale=rationale or "")
                )

            score = 1.0 if not checks else round(1.0 - (unsupported / len(checks)), 2)
            return score, checks
        except Exception:
            # fallback heuristic
            claims = [line[2:].strip() for line in summary.splitlines() if line.startswith("- ")]
            checks: list[ClaimCheck] = []
            source_titles = {document.title.lower() for document in documents}
            unsupported = 0
            for claim in claims:
                supported = any(token.lower() in source_titles for token in claim.split() if len(token) > 4)
                if not supported:
                    unsupported += 1
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        supported=supported,
                        evidence=[document.title for document in documents[:3]] if supported else [],
                        rationale="Matched retrieved source titles" if supported else "No direct source evidence found in the retrieved set",
                    )
                )

            score = 1.0 if not claims else round(1.0 - (unsupported / len(claims)), 2)
            return score, checks
