"""Tests for the planner, report renderer, and history store."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.agents.planner import PlannerAgent
from app.agents.report import ReportAgent, _slugify
from app.models.schemas import ClaimCheck, ResearchRun, SourceDocument
from app.services.history_store import HistoryStore


class TestPlanner:
    def test_strips_conversational_filler_from_search_queries(self):
        plan = PlannerAgent().invoke("Find recent advances in multimodal deepfake detection")
        assert plan.search_queries[0] == "multimodal deepfake detection"

    def test_objective_keeps_the_users_original_wording(self):
        query = "Explain recent advances in neural text generation"
        assert PlannerAgent().invoke(query).objective == query

    def test_search_queries_are_deduplicated(self):
        plan = PlannerAgent().invoke("transformers")
        assert len(plan.search_queries) == len(set(plan.search_queries))

    def test_query_that_is_only_filler_still_produces_a_query(self):
        plan = PlannerAgent().invoke("Explain")
        assert plan.search_queries and plan.search_queries[0]

    def test_steps_are_always_present(self):
        assert len(PlannerAgent().invoke("anything at all").steps) == 5


class TestSlugify:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Find Recent Advances in GANs.", "find-recent-advances-in-gans"),
            ("  multiple   spaces  ", "multiple-spaces"),
            ("!!!", "research-report"),
            ("", "research-report"),
        ],
    )
    def test_slugs(self, text, expected):
        assert _slugify(text) == expected

    def test_slug_is_bounded(self):
        assert len(_slugify("word " * 100)) <= 80


class TestReportAgent:
    @pytest.fixture
    def rendered(self, plan, documents):
        return ReportAgent().invoke(
            query="multimodal deepfake detection",
            plan=plan,
            summary="- Fusion helps.\n- Benchmarks leak.",
            documents=documents,
            critic_score=0.5,
            claim_checks=[
                ClaimCheck(claim="Fusion helps", supported=True, evidence=["[1] Multimodal Fusion"], rationale="Stated in source 1."),
                ClaimCheck(claim="Benchmarks leak", supported=False, evidence=[], rationale="Not addressed."),
            ],
            critic_method="llm",
        )

    def test_markdown_contains_every_section(self, rendered):
        for heading in ("# Research Report", "## Plan", "## Summary", "## Critic Score", "## Sources", "## Claim Checks"):
            assert heading in rendered.markdown

    def test_markdown_labels_the_critic_method(self, rendered):
        assert "verified by the configured LLM" in rendered.markdown

    def test_heuristic_method_is_labelled_as_unverified(self, plan, documents):
        report = ReportAgent().invoke(
            query="q", plan=plan, summary="- a", documents=documents,
            critic_score=1.0, claim_checks=[ClaimCheck(claim="a", supported=True)], critic_method="heuristic",
        )
        assert "no model verified these claims" in report.markdown

    def test_supported_and_unsupported_claims_are_both_shown(self, rendered):
        assert "**SUPPORTED** — Fusion helps" in rendered.markdown
        assert "**UNSUPPORTED** — Benchmarks leak" in rendered.markdown

    def test_source_urls_survive_into_the_markdown(self, rendered, documents):
        for document in documents:
            assert document.url in rendered.markdown

    def test_json_summary_is_serialisable(self, rendered):
        assert json.loads(json.dumps(rendered.json_summary, default=str))["critic_score"] == 0.5

    def test_pdf_is_written_and_is_a_real_pdf(self, rendered):
        assert rendered.pdf_path is not None
        assert Path(rendered.pdf_path).read_bytes()[:5] == b"%PDF-"

    def test_markdown_and_json_artifacts_are_written_beside_the_pdf(self, rendered):
        stem = Path(rendered.pdf_path).with_suffix("")
        assert stem.with_suffix(".md").is_file()
        assert stem.with_suffix(".json").is_file()

    def test_long_lines_are_wrapped_not_truncated(self, plan):
        """Regression: the PDF renderer cut every line at 110 characters, silently
        dropping the tail of long abstracts and URLs."""
        long_url = "https://example.org/" + "segment/" * 40
        report = ReportAgent().invoke(
            query="wrapping test", plan=plan,
            summary="x " * 400,
            documents=[SourceDocument(title="A" * 300, url=long_url, content="c", source="arxiv")],
            critic_score=1.0, claim_checks=[], critic_method="llm",
        )
        assert long_url in report.markdown
        assert Path(report.pdf_path).stat().st_size > 1000

    def test_report_survives_an_unwritable_directory(self, plan, documents, monkeypatch):
        monkeypatch.setattr(
            Path, "write_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("read-only"))
        )
        report = ReportAgent().invoke(
            query="q", plan=plan, summary="- a", documents=documents,
            critic_score=1.0, claim_checks=[], critic_method="llm",
        )
        assert report.markdown, "the in-memory report must still be produced"


class TestHistoryStore:
    def test_append_then_list(self, tmp_path):
        store = HistoryStore(path=tmp_path / "history.jsonl")
        store.append(ResearchRun(query="first"))
        store.append(ResearchRun(query="second"))
        assert {run.query for run in store.list()} == {"first", "second"}

    def test_naive_timestamps_from_older_runs_do_not_break_sorting(self, tmp_path):
        """Regression guard: history written with datetime.utcnow() is naive.
        Sorting naive against timezone-aware values raises TypeError."""
        path = tmp_path / "history.jsonl"
        legacy = {
            "id": "11111111-1111-1111-1111-111111111111",
            "query": "legacy naive run",
            "status": "complete",
            "created_at": "2026-05-30T12:00:00",  # no timezone
            "updated_at": "2026-05-30T12:00:00",
        }
        path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        store = HistoryStore(path=path)
        store.append(ResearchRun(query="modern aware run"))

        runs = store.list()  # must not raise
        assert [run.query for run in runs] == ["modern aware run", "legacy naive run"]
        assert all(run.created_at.tzinfo is not None for run in runs)

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text("{ not json at all\n", encoding="utf-8")
        store = HistoryStore(path=path)
        store.append(ResearchRun(query="good run"))
        assert [run.query for run in store.list()] == ["good run"]

    def test_get_returns_none_for_an_unknown_id(self, tmp_path):
        from uuid import uuid4

        store = HistoryStore(path=tmp_path / "history.jsonl")
        store.append(ResearchRun(query="a run"))
        assert store.get(uuid4()) is None

    def test_get_finds_a_stored_run(self, tmp_path):
        store = HistoryStore(path=tmp_path / "history.jsonl")
        run = ResearchRun(query="findable")
        store.append(run)
        assert store.get(run.id).query == "findable"

    def test_missing_file_lists_empty(self, tmp_path):
        assert HistoryStore(path=tmp_path / "nope" / "history.jsonl").list() == []

    def test_limit_is_applied(self, tmp_path):
        store = HistoryStore(path=tmp_path / "history.jsonl")
        for index in range(5):
            store.append(ResearchRun(query=f"run {index}"))
        assert len(store.list(limit=2)) == 2


class TestSchemas:
    def test_utc_now_is_timezone_aware(self):
        from app.models.schemas import utc_now

        now = utc_now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_new_runs_carry_aware_timestamps(self):
        assert ResearchRun(query="q").created_at.tzinfo is not None
