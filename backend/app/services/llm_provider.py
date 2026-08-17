"""LLM provider abstraction for the summarizer and critic agents.

Two public entry points:

* :func:`summarize_with_provider` — synthesises retrieved sources into a Markdown
  summary followed by a ``===JSON===`` block of claim checks.
* :func:`critique_with_provider` — grades an *already written* summary against the
  retrieved sources and returns only the ``===JSON===`` claim-check block.

Both are synchronous and blocking. Callers on an event loop must offload them to a
worker thread (``asyncio.to_thread``); ``app.graph.workflow`` does this for every
node so the WebSocket handler never stalls.

Both degrade to a deterministic local fallback when no provider is configured, no
API key is present, or every provider call fails. The fallback is always labelled
as such so a reader can tell a real model run from a stub.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, List

from app.core.config import settings

JSON_MARKER = "===JSON==="

_LOGS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs"
)

# Text the local fallback stamps into its claim check so downstream consumers
# (and the reader of a report) can tell it apart from a real model verdict.
FALLBACK_CLAIM = "Summary produced by the local deterministic fallback (no LLM was used)"


# Most recent provider failure, surfaced by GET /health.
#
# This exists because "a key is set" and "the provider works" are different states,
# and the gap between them is invisible: an invalid or rate-limited key makes every
# summary come from the local fallback while health still reports configured. Without
# this, diagnosing a deployment means reading log files on the host.
_last_error: str | None = None


def last_provider_error() -> str | None:
    return _last_error


def _log(filename: str, message: str) -> None:
    """Best-effort append to backend/logs/<filename>. Never raises."""
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        with open(os.path.join(_LOGS_DIR, filename), "a", encoding="utf-8") as handle:
            handle.write(message.rstrip("\n") + "\n")
    except Exception:
        pass


def _trace(message: str) -> None:
    _log("llm_provider_trace.log", message)


def _error(message: str) -> None:
    global _last_error
    _last_error = message[:500]
    _log("llm_provider_errors.log", message)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    openai_key: str | None
    gemini_key: str | None

    @property
    def usable(self) -> bool:
        if self.provider == "openai":
            return bool(self.openai_key)
        if self.provider == "gemini":
            return bool(self.gemini_key)
        return False


def resolve_provider_config(model_name: str | None = None) -> ProviderConfig:
    """Read provider settings, preferring live environment variables over the
    cached ``settings`` snapshot so tests can override them at runtime."""
    provider = (os.getenv("DEFAULT_LLM_PROVIDER") or settings.default_llm_provider or "mock").strip().lower()
    model = (model_name or os.getenv("DEFAULT_LLM_MODEL") or settings.default_llm_model or "").strip()

    if not model:
        model = "gpt-4o-mini" if provider == "openai" else "gemini-2.0-flash"

    config = ProviderConfig(
        provider=provider,
        model=model,
        openai_key=os.getenv("OPENAI_API_KEY") or settings.openai_api_key,
        gemini_key=os.getenv("GEMINI_API_KEY") or settings.gemini_api_key,
    )
    _trace(
        f"resolve provider={config.provider} model={config.model} "
        f"openai_key_set={bool(config.openai_key)} gemini_key_set={bool(config.gemini_key)}"
    )
    return config


def _prepare_snippets(documents: List[Any], max_chars: int = 800, max_documents: int = 8) -> str:
    """Render documents as a numbered source list the model can cite by index."""
    parts = []
    for index, document in enumerate(documents[:max_documents], start=1):
        title = getattr(document, "title", "") or ""
        url = getattr(document, "url", None) or ""
        abstract = getattr(document, "abstract", None) or ""
        content = getattr(document, "content", "") or ""
        snippet = (abstract or content)[:max_chars]
        parts.append(f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}\n")
    return "\n".join(parts)


def parse_json_section(text: str) -> tuple[str, dict]:
    """Split ``text`` at the ``===JSON===`` marker and parse the JSON half.

    Returns ``(markdown, data)``. ``data`` is ``{}`` when no JSON could be located,
    or ``{"error": "failed_to_parse_json", ...}`` when a candidate was found but
    would not parse — callers must check for that key before trusting the payload.
    """
    if not text:
        return "", {}

    def _loads(candidate: str) -> dict | None:
        candidate = candidate.strip()
        # Models often wrap JSON in a ```json fence despite instructions.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
        if match:
            trimmed = match.group(1)
            try:
                return json.loads(trimmed)
            except Exception:
                try:
                    # Tolerate Python literals (True/False/None, single quotes).
                    return json.loads(json.dumps(ast.literal_eval(trimmed)))
                except Exception:
                    return None
        return None

    if JSON_MARKER in text:
        markdown, _, json_text = text.partition(JSON_MARKER)
        data = _loads(json_text)
        if data is None:
            return markdown.strip(), {"error": "failed_to_parse_json", "raw": json_text.strip()}
        return markdown.strip(), data

    # No marker: fall back to the last JSON object in the text.
    match = re.search(r"(\{[\s\S]*\})\s*$", text)
    if match:
        data = _loads(match.group(1))
        if data is None:
            return text.strip(), {"error": "failed_to_parse_json", "raw": match.group(1)}
        return text[: match.start()].strip(), data

    return text.strip(), {}


# Kept as the historical public name used by the agents.
try_parse_json_section = parse_json_section


def _has_usable_json(text: str | None) -> bool:
    if not text:
        return False
    _markdown, data = parse_json_section(text)
    return bool(data) and not data.get("error")


# --------------------------------------------------------------------------- #
# Provider calls
# --------------------------------------------------------------------------- #


def _call_openai(prompt: str, config: ProviderConfig) -> str | None:
    """Call the OpenAI Chat Completions API using the v1+ SDK surface."""
    try:
        from openai import OpenAI
    except Exception as exc:
        _error(f"openai SDK unavailable: {type(exc).__name__}: {exc}")
        return None

    try:
        client = OpenAI(api_key=config.openai_key)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a careful research summarisation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1600,
        )
        text = (response.choices[0].message.content or "").strip()
        _trace(f"openai chat.completions model={config.model} chars={len(text)}")
        return text or None
    except Exception as exc:
        _error(f"openai call failed: {type(exc).__name__}: {exc}")
        return None


def _gemini_model_name(model: str) -> str:
    """google-genai accepts either ``gemini-2.0-flash`` or ``models/gemini-2.0-flash``.

    Normalise to the bare name, which both the current and legacy SDKs accept.
    """
    return model[len("models/") :] if model.startswith("models/") else model


def _call_gemini(prompt: str, config: ProviderConfig) -> str | None:
    """Call Gemini via google-genai (preferred) or the legacy google-generativeai."""
    model = _gemini_model_name(config.model)

    # Preferred: google-genai >= 1.0 Client API.
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=config.gemini_key)
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(response, "text", None) or "").strip()
            _trace(f"gemini genai.Client model={model} chars={len(text)}")
            if text:
                return text
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception as exc:
        _error(f"gemini google-genai call failed: {type(exc).__name__}: {exc}")

    # Legacy: google-generativeai GenerativeModel API.
    try:
        import google.generativeai as legacy_genai  # type: ignore

        legacy_genai.configure(api_key=config.gemini_key)
        response = legacy_genai.GenerativeModel(model).generate_content(prompt)
        text = (getattr(response, "text", None) or "").strip()
        _trace(f"gemini legacy GenerativeModel model={model} chars={len(text)}")
        return text or None
    except Exception as exc:
        _error(f"gemini google-generativeai call failed: {type(exc).__name__}: {exc}")
        return None


_JSON_ONLY_NUDGE = (
    "\n\nYour previous response did not contain a parseable JSON object. "
    f"Respond again with a line containing exactly {JSON_MARKER} followed by the JSON object "
    "and nothing else. No prose, no markdown code fences."
)


def _generate(prompt: str, config: ProviderConfig, attempts: int = 2) -> str | None:
    """Call the configured provider, retrying with a JSON-only nudge if needed.

    ``attempts`` is the total number of calls, so the default of 2 means one
    initial call plus one retry.
    """
    if not config.usable:
        _trace(f"provider {config.provider!r} not usable (missing key or mock) — using local fallback")
        return None

    callers: dict[str, Callable[[str, ProviderConfig], str | None]] = {
        "openai": _call_openai,
        "gemini": _call_gemini,
    }
    call = callers.get(config.provider)
    if call is None:
        _trace(f"unknown provider {config.provider!r} — using local fallback")
        return None

    best: str | None = None
    for attempt in range(max(1, attempts)):
        current_prompt = prompt if attempt == 0 else prompt + _JSON_ONLY_NUDGE
        text = call(current_prompt, config)
        if text:
            best = best or text
            if _has_usable_json(text):
                return text
            _trace(f"attempt {attempt} returned text without usable JSON")
        if attempt + 1 < max(1, attempts):
            time.sleep(min(2**attempt, 4))

    # Return whatever prose we got: the caller still shows it, and the JSON
    # section simply comes back empty rather than pretending to be verified.
    return best


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_CLAIM_SCHEMA = (
    "The JSON must be a single top-level object with the key `claim_checks` whose value is an "
    "array of objects. Each object must have: `claim` (string), `supported` (true/false), "
    "`evidence` (array of integer source indices from the numbered Sources above), "
    "`evidence_snippets` (array of short verbatim strings from those cited sources), and "
    "`rationale` (string).\n"
    "Use only source indices that appear in the Sources list. Never invent a source. If you "
    "cannot find direct supporting evidence for a claim, set `supported` to false and leave "
    "`evidence` as an empty array."
)


def _summarize_prompt(objective: str, snippets: str) -> str:
    return (
        f"Research objective:\n{objective}\n\n"
        f"Sources:\n{snippets}\n\n"
        "Instruction:\n"
        "You are a careful research assistant. Using ONLY the numbered sources above, produce "
        "exactly two parts in this order:\n\n"
        "1) A short Markdown summary (3-6 bullet points) synthesising the main approaches, "
        "findings, and open problems.\n\n"
        f"2) A line containing exactly {JSON_MARKER} on its own, followed immediately by a JSON "
        "object and nothing else. Extract the claims you made in part 1 and check each one. "
        f"{_CLAIM_SCHEMA}\n"
    )


def _critique_prompt(summary: str, snippets: str) -> str:
    return (
        f"Sources:\n{snippets}\n\n"
        f"Summary under review:\n{summary}\n\n"
        "Instruction:\n"
        "You are a strict fact-checker. Break the summary under review into its individual "
        "factual claims, then verify each claim against the numbered sources above. Judge only "
        "what the sources actually support — do not use outside knowledge, and do not give a "
        "claim the benefit of the doubt.\n\n"
        f"Respond with a line containing exactly {JSON_MARKER} followed by a JSON object and "
        f"nothing else. {_CLAIM_SCHEMA}\n"
    )


# --------------------------------------------------------------------------- #
# Local fallbacks
# --------------------------------------------------------------------------- #


def _fallback_summary(objective: str, documents: List[Any]) -> str:
    if not documents:
        markdown = "- No source documents were retrieved for this objective."
    else:
        sources = sorted({getattr(document, "source", "unknown") for document in documents})
        years = sorted({y for y in (getattr(d, "year", None) for d in documents) if y})
        lines = [
            f"- Research objective: {objective}",
            f"- Retrieved {len(documents)} source documents from: {', '.join(sources)}.",
        ]
        if years:
            lines.append(f"- Publication years observed: {years[0]}–{years[-1]}.")
        # "not configured" and "configured but every call failed" both land here, so
        # the wording must cover both rather than asserting the first.
        lines.append(
            "- No working LLM provider was available, so this listing is mechanical: it reports "
            "what was retrieved without synthesising it. See GET /health for the reason."
        )
        lines.append("- Representative sources:")
        lines.extend(f"  - {getattr(document, 'title', 'Untitled')}" for document in documents[:6])
        markdown = "\n".join(lines)

    placeholder = {
        # Consumers key off this to label the run as unverified rather than
        # inferring it from the text.
        "method": "heuristic",
        "claim_checks": [
            {
                "claim": FALLBACK_CLAIM,
                "supported": False,
                "evidence": [],
                "evidence_snippets": [],
                "rationale": "No working LLM provider was available, so no claim was verified.",
            }
        ],
    }
    return markdown + f"\n\n{JSON_MARKER}\n" + json.dumps(placeholder)


def _fallback_claim_checks(summary: str, documents: List[Any]) -> str:
    """Heuristic critique: match summary bullets against source titles.

    Deliberately conservative — a keyword overlap is weak evidence, so the
    rationale says so rather than claiming verification.
    """
    bullets = [
        line.lstrip("-* ").strip()
        for line in (summary or "").splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    title_tokens = {
        token.lower()
        for document in documents
        for token in re.findall(r"\w+", getattr(document, "title", "") or "")
        if len(token) > 4
    }

    checks = []
    for bullet in bullets:
        tokens = {t.lower() for t in re.findall(r"\w+", bullet) if len(t) > 4}
        overlap = sorted(tokens & title_tokens)
        checks.append(
            {
                "claim": bullet,
                "supported": bool(overlap),
                "evidence": [],
                "evidence_snippets": overlap[:5],
                "rationale": (
                    f"Heuristic keyword overlap with retrieved source titles ({', '.join(overlap[:5])}). "
                    "This is a weak signal, not model verification."
                    if overlap
                    else "No keyword overlap with any retrieved source title."
                ),
            }
        )
    return f"{JSON_MARKER}\n" + json.dumps({"method": "heuristic", "claim_checks": checks})


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def summarize_with_provider(plan: Any, documents: List[Any], model_name: str | None = None) -> str:
    """Return ``markdown + ===JSON=== + claim_checks`` for the retrieved documents."""
    objective = ""
    if plan is not None:
        objective = getattr(plan, "objective", None) or str(plan)

    snippets = _prepare_snippets(documents)
    config = resolve_provider_config(model_name)

    text = _generate(_summarize_prompt(objective, snippets), config)
    if text:
        return text
    return _fallback_summary(objective, documents)


def critique_with_provider(summary: str, documents: List[Any], model_name: str | None = None) -> str:
    """Grade ``summary`` — the exact text the user sees — against ``documents``.

    Returns a ``===JSON===`` block of claim checks. This is a separate call from
    :func:`summarize_with_provider` on purpose: the critic must review the summary
    that was actually published, not an independently redrafted one.
    """
    snippets = _prepare_snippets(documents)
    config = resolve_provider_config(model_name)

    text = _generate(_critique_prompt(summary, snippets), config)
    if text and _has_usable_json(text):
        return text
    return _fallback_claim_checks(summary, documents)
