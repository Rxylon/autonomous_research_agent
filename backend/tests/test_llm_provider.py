"""Tests for the provider abstraction, including regressions for two shipped bugs."""

from __future__ import annotations

import sys

import pytest

from app.services import llm_provider
from app.services.llm_provider import (
    FALLBACK_CLAIM,
    JSON_MARKER,
    ProviderConfig,
    _generate,
    critique_with_provider,
    parse_json_section,
    resolve_provider_config,
    summarize_with_provider,
)


class TestParseJsonSection:
    def test_splits_on_marker(self):
        markdown, data = parse_json_section(f'- a bullet\n{JSON_MARKER}\n{{"claim_checks": []}}')
        assert markdown == "- a bullet"
        assert data == {"claim_checks": []}

    def test_tolerates_code_fence_around_json(self):
        text = f'- bullet\n{JSON_MARKER}\n```json\n{{"claim_checks": [1]}}\n```'
        _markdown, data = parse_json_section(text)
        assert data == {"claim_checks": [1]}

    def test_tolerates_python_literals(self):
        text = f"{JSON_MARKER}\n{{'claim_checks': [{{'supported': True}}]}}"
        _markdown, data = parse_json_section(text)
        assert data["claim_checks"][0]["supported"] is True

    def test_finds_trailing_json_without_marker(self):
        markdown, data = parse_json_section('some prose\n{"claim_checks": []}')
        assert markdown == "some prose"
        assert data == {"claim_checks": []}

    def test_flags_unparseable_json_rather_than_returning_empty(self):
        _markdown, data = parse_json_section(f"{JSON_MARKER}\nnot json at all")
        assert data.get("error") == "failed_to_parse_json"

    def test_no_json_returns_empty_dict(self):
        markdown, data = parse_json_section("just a summary")
        assert markdown == "just a summary"
        assert data == {}

    def test_empty_input(self):
        assert parse_json_section("") == ("", {})


class TestProviderConfig:
    def test_mock_provider_is_not_usable(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "mock")
        assert resolve_provider_config().usable is False

    def test_provider_without_key_is_not_usable(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = resolve_provider_config()
        assert config.provider == "gemini"
        assert config.usable is False

    def test_provider_with_key_is_usable(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert resolve_provider_config().usable is True

    def test_explicit_model_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_LLM_MODEL", "from-env")
        assert resolve_provider_config("explicit-model").model == "explicit-model"


class TestGenerateRetry:
    """Regression: `_request_json_only_gemini` rebound its own `attempts: int`
    parameter to a list, so `range(max(1, attempts))` raised TypeError comparing
    int to list. A broad `except Exception` swallowed it, silently costing the
    retry path. `_generate` must be able to loop.
    """

    def test_retries_until_json_appears(self, monkeypatch):
        calls: list[str] = []

        def fake_call(prompt: str, config: ProviderConfig) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "prose only, no json"
            return f'{JSON_MARKER}\n{{"claim_checks": []}}'

        monkeypatch.setattr(llm_provider, "_call_gemini", fake_call)
        monkeypatch.setattr(llm_provider.time, "sleep", lambda _seconds: None)

        config = ProviderConfig(provider="gemini", model="m", openai_key=None, gemini_key="k")
        result = _generate("base prompt", config, attempts=2)

        assert len(calls) == 2, "the retry never ran"
        assert "did not contain a parseable JSON object" in calls[1]
        assert parse_json_section(result)[1] == {"claim_checks": []}

    def test_stops_early_when_first_call_is_good(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            llm_provider,
            "_call_gemini",
            lambda p, c: calls.append(p) or f'{JSON_MARKER}\n{{"claim_checks": []}}',
        )
        config = ProviderConfig(provider="gemini", model="m", openai_key=None, gemini_key="k")
        _generate("prompt", config, attempts=3)
        assert len(calls) == 1

    def test_returns_prose_when_json_never_arrives(self, monkeypatch):
        monkeypatch.setattr(llm_provider, "_call_gemini", lambda p, c: "only prose")
        monkeypatch.setattr(llm_provider.time, "sleep", lambda _s: None)
        config = ProviderConfig(provider="gemini", model="m", openai_key=None, gemini_key="k")
        assert _generate("prompt", config, attempts=2) == "only prose"

    def test_unusable_config_skips_provider_entirely(self, monkeypatch):
        monkeypatch.setattr(
            llm_provider, "_call_gemini", lambda p, c: pytest.fail("provider must not be called")
        )
        config = ProviderConfig(provider="gemini", model="m", openai_key=None, gemini_key=None)
        assert _generate("prompt", config) is None


class TestFallbacks:
    def test_summary_fallback_labels_itself(self, documents, plan):
        raw = summarize_with_provider(plan, documents)
        markdown, data = parse_json_section(raw)
        assert data["method"] == "heuristic"
        assert data["claim_checks"][0]["claim"] == FALLBACK_CLAIM
        assert data["claim_checks"][0]["supported"] is False
        assert "No working LLM provider was available" in markdown

    def test_summary_fallback_with_no_documents(self, plan):
        markdown, _data = parse_json_section(summarize_with_provider(plan, []))
        assert "No source documents" in markdown

    def test_critique_fallback_is_marked_heuristic(self, documents):
        summary = "- Multimodal fusion improves detection\n- Unrelated tangent about quantum tunnelling"
        _markdown, data = parse_json_section(critique_with_provider(summary, documents))
        assert data["method"] == "heuristic"
        checks = data["claim_checks"]
        assert len(checks) == 2
        assert checks[0]["supported"] is True, "shares 'multimodal'/'fusion'/'detection' with a title"
        assert checks[1]["supported"] is False, "shares nothing with any source title"

    def test_critique_fallback_on_summary_with_no_bullets(self, documents):
        _markdown, data = parse_json_section(critique_with_provider("A prose paragraph.", documents))
        assert data["claim_checks"] == []


class TestCritiqueUsesTheGivenSummary:
    """Regression: the critic used to call `summarize_with_provider(None, documents)`,
    grading an independently redrafted summary instead of the published one.
    """

    def test_summary_text_is_in_the_prompt(self, documents, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(llm_provider, "_call_gemini", lambda p, c: seen.append(p) or f'{JSON_MARKER}\n{{"claim_checks": []}}')
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "k")

        critique_with_provider("- A very distinctive claim about sensor drift", documents)

        assert len(seen) == 1
        assert "A very distinctive claim about sensor drift" in seen[0]
        assert "Summary under review" in seen[0]


class TestProviderErrorTracking:
    def test_a_failed_call_is_recorded_for_health(self, monkeypatch, documents, plan):
        """Regression: provider failures went only to a log file on the host, so a
        deployment silently serving fallback summaries looked healthy."""
        monkeypatch.setattr(llm_provider, "_last_error", None)
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")
        monkeypatch.setattr(llm_provider.time, "sleep", lambda _s: None)

        def exploding_client(*_args, **_kwargs):
            raise RuntimeError("400 API_KEY_INVALID")

        # Force both SDK paths to fail.
        monkeypatch.setitem(sys.modules, "google.generativeai", None)
        monkeypatch.setattr(llm_provider, "_call_gemini", lambda prompt, config: llm_provider._error("gemini call failed: 400 API_KEY_INVALID") or None)

        summarize_with_provider(plan, documents)
        assert "API_KEY_INVALID" in (llm_provider.last_provider_error() or "")

    def test_error_text_is_bounded(self, monkeypatch):
        monkeypatch.setattr(llm_provider, "_last_error", None)
        llm_provider._error("x" * 5000)
        assert len(llm_provider.last_provider_error()) == 500


class TestSnippetPreparation:
    def test_numbers_sources_from_one(self, documents):
        snippets = llm_provider._prepare_snippets(documents)
        assert snippets.startswith("[1] Multimodal Fusion")
        assert "[2] Temporal Consistency" in snippets

    def test_caps_document_count_and_snippet_length(self, documents):
        many = documents * 10
        snippets = llm_provider._prepare_snippets(many, max_chars=10, max_documents=3)
        assert snippets.count("URL:") == 3
