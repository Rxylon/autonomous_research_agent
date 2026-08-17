"""Tests for the critic, centred on the score-inflation regression."""

from __future__ import annotations

import json

from app.agents.critic import CriticAgent
from app.models.schemas import SourceDocument
from app.services import llm_provider
from app.services.llm_provider import JSON_MARKER


def _stub_verdict(monkeypatch, claim_checks, method=None):
    """Make `critique_with_provider` return a fixed claim-check payload."""
    payload = {"claim_checks": claim_checks}
    if method:
        payload["method"] = method
    monkeypatch.setattr(
        llm_provider,
        "critique_with_provider",
        lambda summary, documents, model_name=None: f"{JSON_MARKER}\n{json.dumps(payload)}",
    )


class TestSupportedVerdictIsRespected:
    """Regression: `if evidence_list: supported = True` overrode the model's own
    `supported: false`, so any claim with an attached snippet counted as verified
    and scores drifted toward 1.0.
    """

    def test_unsupported_claim_with_evidence_stays_unsupported(self, documents, monkeypatch):
        _stub_verdict(
            monkeypatch,
            [
                {
                    "claim": "Fusion solves deepfake detection entirely",
                    "supported": False,
                    "evidence": [1],
                    "evidence_snippets": ["We study audio-visual fusion"],
                    "rationale": "The source discusses fusion but claims no such result.",
                }
            ],
        )
        verdict = CriticAgent().invoke("- a claim", documents)

        assert verdict.claim_checks[0].supported is False
        assert verdict.score == 0.0
        assert verdict.claim_checks[0].evidence, "evidence is still surfaced to the reader"

    def test_score_is_the_supported_fraction(self, documents, monkeypatch):
        _stub_verdict(
            monkeypatch,
            [
                {"claim": "a", "supported": True, "evidence": [1]},
                {"claim": "b", "supported": False, "evidence": []},
                {"claim": "c", "supported": True, "evidence": [2]},
                {"claim": "d", "supported": False, "evidence": [1]},
            ],
        )
        assert CriticAgent().invoke("- s", documents).score == 0.5

    def test_all_supported_scores_one(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": [1]}])
        assert CriticAgent().invoke("- s", documents).score == 1.0


class TestEvidenceResolution:
    def test_valid_index_becomes_a_citation(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": [2]}])
        evidence = CriticAgent().invoke("- s", documents).claim_checks[0].evidence
        assert evidence[0].startswith("[2] Temporal Consistency Cues in Video Forensics")

    def test_out_of_range_index_is_dropped_not_clamped(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": [9, 0, -1]}])
        assert CriticAgent().invoke("- s", documents).claim_checks[0].evidence == []

    def test_non_integer_indices_are_ignored(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": ["1", None, 1.5]}])
        assert CriticAgent().invoke("- s", documents).claim_checks[0].evidence == []

    def test_snippets_are_truncated(self, documents, monkeypatch):
        _stub_verdict(
            monkeypatch,
            [{"claim": "a", "supported": True, "evidence": [], "evidence_snippets": ["x" * 5000]}],
        )
        evidence = CriticAgent().invoke("- s", documents).claim_checks[0].evidence
        assert len(evidence[0]) == 300

    def test_duplicate_evidence_is_deduplicated(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": [1, 1, 1]}])
        assert len(CriticAgent().invoke("- s", documents).claim_checks[0].evidence) == 1


class TestMethodReporting:
    def test_llm_verdict_is_labelled_llm(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True, "evidence": [1]}])
        assert CriticAgent().invoke("- s", documents).method == "llm"

    def test_heuristic_verdict_is_labelled_heuristic(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": True}], method="heuristic")
        assert CriticAgent().invoke("- s", documents).method == "heuristic"

    def test_no_provider_configured_falls_back_to_heuristic(self, documents):
        """With DEFAULT_LLM_PROVIDER=mock (see conftest) no model runs, so the
        score must be labelled heuristic rather than presented as verification."""
        verdict = CriticAgent().invoke("- Multimodal fusion improves detection", documents)
        assert verdict.method == "heuristic"


class TestDegenerateInputs:
    def test_empty_summary_scores_zero(self, documents):
        verdict = CriticAgent().invoke("", documents)
        assert verdict.score == 0.0
        assert verdict.method == "empty"

    def test_whitespace_summary_scores_zero(self, documents):
        assert CriticAgent().invoke("   \n  ", documents).method == "empty"

    def test_no_claim_checks_scores_zero_not_one(self, documents, monkeypatch):
        """An empty claim list means nothing was checked. Scoring that 1.0 — as the
        old code did — reports perfect support for zero evidence."""
        _stub_verdict(monkeypatch, [])
        verdict = CriticAgent().invoke("- s", documents)
        assert verdict.score == 0.0
        assert verdict.method == "empty"

    def test_unparseable_provider_output_scores_zero(self, documents, monkeypatch):
        monkeypatch.setattr(
            llm_provider,
            "critique_with_provider",
            lambda summary, documents, model_name=None: f"{JSON_MARKER}\ngarbage",
        )
        assert CriticAgent().invoke("- s", documents).score == 0.0

    def test_bare_string_claims_are_not_counted_as_supported(self, documents, monkeypatch):
        _stub_verdict(monkeypatch, ["just a string claim"])
        verdict = CriticAgent().invoke("- s", documents)
        assert verdict.claim_checks[0].supported is False
        assert verdict.score == 0.0

    def test_claims_with_blank_text_are_skipped(self, documents, monkeypatch):
        _stub_verdict(
            monkeypatch,
            [{"claim": "  ", "supported": True}, {"claim": "real", "supported": True}],
        )
        verdict = CriticAgent().invoke("- s", documents)
        assert [c.claim for c in verdict.claim_checks] == ["real"]

    def test_works_with_no_documents(self, monkeypatch):
        _stub_verdict(monkeypatch, [{"claim": "a", "supported": False, "evidence": [1]}])
        verdict = CriticAgent().invoke("- s", [])
        assert verdict.score == 0.0
        assert verdict.claim_checks[0].evidence == []
