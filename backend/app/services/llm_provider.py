from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from app.core.config import settings


def _prepare_snippets(documents: List[Any], max_chars: int = 800) -> str:
    parts = []
    for i, doc in enumerate(documents[:8], start=1):
        title = getattr(doc, "title", "")
        url = getattr(doc, "url", None) or ""
        abstract = getattr(doc, "abstract", None) or ""
        content = getattr(doc, "content", "") or ""
        snippet = abstract or (content[:max_chars] if content else "")
        # include an explicit index, title, url, and short snippet
        parts.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
    return "\n".join(parts)


def summarize_with_provider(plan: Any, documents: List[Any], model_name: str | None = None) -> str:
    try:
        model = model_name or os.getenv("DEFAULT_LLM_MODEL") or "gpt-3.5-turbo"
        # prefer runtime environment variables (tests may load .env at runtime)
        provider = os.getenv("DEFAULT_LLM_PROVIDER") or settings.default_llm_provider or "mock"
        openai_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        gemini_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
        # determine repository logs directory (absolute) and trace runtime resolution
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, "logs")
        try:
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                fh.write(f"ENTRY provider={provider} openai_key_set={bool(openai_key)} gemini_key_set={bool(gemini_key)} model={model}\n")
        except Exception:
            pass
    except Exception as exc:
        # if we fail to even determine config, log to stderr and fall back to mock
        print(f"Error determining LLM provider configuration: {type(exc).__name__}: {exc}")
        provider = "mock"

    prompt = (
        "You are a careful research assistant. Given the research objective and a set of numbered source snippets, produce the following OUTPUT STRICTLY and in this order:\n"
        "\n1) A short Markdown summary (3-6 bullet points) that synthesizes main approaches, findings, and open problems.\n"
        "\n2) A JSON object delimited by a single line containing exactly ===JSON=== (three equal signs, no extra text) on its own line, followed immediately by the JSON text and nothing else. The JSON must be a single top-level object with the key `claim_checks` whose value is an array of objects. Each claim object must have the fields: `claim` (string), `supported` (true/false), `evidence` (array of integer source indices referencing the numbered Sources above), `evidence_snippets` (array of short strings drawn from the cited sources), and `rationale` (string).\n"
        "\nIMPORTANT: Use only the source indices present in the Sources list. Do not invent sources. If you cannot find direct supporting evidence for a claim, set `supported` to false and set `evidence` to an empty array.\n"
        "\nAlways follow the marker format exactly: produce the Markdown summary, then a line with ===JSON===, then the JSON object only.\n\n"
    )

    snippets = _prepare_snippets(documents)
    objective = getattr(plan, "objective", str(plan)) if plan else ""

    # if using Gemini and no Gemini-specific model was provided, prefer a Gemini model
    if provider == "gemini" and (not model or model == "gpt-3.5-turbo"):
        model = os.getenv("DEFAULT_LLM_MODEL") or "models/gemini-1.0"

    full_prompt = (
        f"Research objective:\n{objective}\n\nSources:\n{snippets}\n\nInstruction:\n{prompt}"
    )

    def _extract_json(text: str) -> tuple[str, dict]:
        marker = "===JSON==="
        if marker in text:
            parts = text.split(marker, 1)
            md = parts[0].strip()
            json_text = parts[1].strip()
            # try strict json first
            try:
                data = json.loads(json_text)
                return md, data
            except Exception:
                # attempt tolerant parse using ast.literal_eval after some cleanup
                cleaned = json_text.strip()
                # remove leading/trailing non-json
                m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
                if m:
                    cleaned = m.group(1)
                try:
                    data = ast.literal_eval(cleaned)
                    # convert Python literals to JSON-serializable dict
                    json_str = json.dumps(data)
                    data = json.loads(json_str)
                    return md, data
                except Exception:
                    return md, {"error": "failed_to_parse_json", "raw": json_text}
        # no marker
        # also try to find a JSON object at end of text
        m = re.search(r"(\{[\s\S]*\})\s*$", text)
        if m:
            md = text[: m.start()].strip()
            json_text = m.group(1)
            try:
                data = json.loads(json_text)
                return md, data
            except Exception:
                try:
                    data = ast.literal_eval(json_text)
                    json_str = json.dumps(data)
                    data = json.loads(json_str)
                    return md, data
                except Exception:
                    return text, {"error": "failed_to_parse_json", "raw": json_text}
        return text, {}

    # helper to request only JSON if the full response failed to include valid JSON
    def _request_json_only_openai(chat_messages: list[dict], attempts: int = 1) -> str | None:
        try:
            import openai
            openai.api_key = settings.openai_api_key
            followup = (
                "The previous response did not contain a valid JSON object.\n"
                "Strictly respond with exactly one line that starts with ===JSON=== followed by the JSON object and nothing else. No markdown, no explanation."
            )
            for i in range(max(1, attempts)):
                messages = chat_messages + [{"role": "user", "content": followup}]
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1200,
                    )
                    text = resp["choices"][0]["message"]["content"].strip()
                    # quick validation
                    if "===JSON===" in text or re.search(r"\{[\s\S]*\}", text):
                        return text
                except Exception:
                    pass
                # exponential backoff small sleep
                time.sleep(min(2 ** i, 4))
            return None
        except Exception:
            return None

    def _request_json_only_gemini(base_prompt: str, attempts: int = 2) -> str | None:
        try:
            # prefer the new package name if available, fall back to the deprecated one
            try:
                import google.genai as genai
            except Exception:
                import google.generativeai as genai

            # configure client
            try:
                genai.configure(api_key=settings.gemini_api_key)
            except Exception:
                # some genai variants may not expose configure; set environment variable instead
                os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
            followup = (
                "The previous response did not contain a valid JSON object.\n"
                "Please respond with exactly one line containing ===JSON=== followed by the JSON object only (no markdown, no explanation)."
            )
            chosen_model = model
            if not chosen_model.startswith("models/"):
                chosen_model = f"models/{chosen_model}" if not chosen_model.startswith("gemini") else f"models/{chosen_model}"

            # Prefer the `Client` API if available (google.genai>=2.x)
            genai_dir = dir(genai)
            attempts = []

            def _extract_text_from_resp(resp):
                if resp is None:
                    return ""
                if isinstance(resp, str):
                    return resp
                for attr in ("text", "result", "output", "data"):
                    if hasattr(resp, attr):
                        val = getattr(resp, attr)
                        if isinstance(val, str):
                            return val
                        try:
                            if isinstance(val, (list, tuple)) and len(val) > 0:
                                first = val[0]
                                if isinstance(first, str):
                                    return first
                                if hasattr(first, "text"):
                                    return first.text
                                if hasattr(first, "content"):
                                    return first.content
                        except Exception:
                            pass
                        try:
                            if isinstance(val, dict) and "text" in val:
                                return val["text"]
                        except Exception:
                            pass
                try:
                    return str(resp)
                except Exception:
                    return ""

            # Try to instantiate the genai Client with the API key and call models.generate_content
            try:
                ClientClass = getattr(genai, 'Client', None) or getattr(getattr(genai, 'client', {}), 'Client', None)
                if ClientClass:
                    try:
                        client = ClientClass(api_key=settings.gemini_api_key)
                    except Exception:
                        # fallback: try passing api_key as positional or via environment
                        os.environ.setdefault('GEMINI_API_KEY', settings.gemini_api_key)
                        client = ClientClass()
                    try:
                        if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
                            resp = client.models.generate_content(model=chosen_model, contents=base_prompt + "\n\n" + followup)
                            text = _extract_text_from_resp(getattr(resp, 'text', resp))
                            attempts.append(('client.models.generate_content', 'ok'))
                            try:
                                os.makedirs(logs_dir, exist_ok=True)
                                with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                                    fh.write(f"Gemini used client.models.generate_content resp_type={type(resp)}\n")
                            except Exception:
                                pass
                            try:
                                client.close()
                            except Exception:
                                pass
                            return text
                    except Exception as exc:
                        attempts.append(('client.models.generate_content', f'exc:{type(exc).__name__}'))
                        try:
                            os.makedirs(logs_dir, exist_ok=True)
                            with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                                fh.write(f"Gemini client.models.generate_content failed: {type(exc).__name__}: {exc}\n")
                        except Exception:
                            pass

            except Exception:
                pass

            # Fallback: try legacy top-level procedural call surfaces we might have missed
            # (keep previous defensive attempts for broad compatibility)
            call_candidates = [
                ("generate_text", lambda: genai.generate_text(model=chosen_model, prompt=base_prompt + "\n\n" + followup, temperature=0.0) if hasattr(genai, "generate_text") else None),
                ("Text.generate", lambda: genai.Text.generate(model=chosen_model, prompt=base_prompt + "\n\n" + followup, temperature=0.0) if hasattr(genai, "Text") and hasattr(genai.Text, "generate") else None),
                ("generate", lambda: genai.generate(model=chosen_model, prompt=base_prompt + "\n\n" + followup, temperature=0.0) if hasattr(genai, "generate") else None),
                ("chat.create", lambda: genai.chat.create(model=chosen_model, messages=[{"role": "user", "content": base_prompt + "\n\n" + followup}], temperature=0.0) if hasattr(genai, "chat") and hasattr(genai.chat, "create") else None),
                ("chats.create", lambda: genai.chats.create(model=chosen_model, messages=[{"role": "user", "content": base_prompt + "\n\n" + followup}], temperature=0.0) if hasattr(genai, "chats") and hasattr(genai.chats, "create") else None),
            ]

            # Try multiple passes of fallback call candidates to increase robustness
            for attempt_index in range(max(1, attempts)):
                for name, fn in call_candidates:
                    try:
                        if fn is None:
                            attempts.append((name, "not_available"))
                            continue
                        resp = fn()
                        text = _extract_text_from_resp(resp)
                        attempts.append((name, "ok"))
                        try:
                            os.makedirs(logs_dir, exist_ok=True)
                            with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                                fh.write(f"Gemini used method={name} resp_type={type(resp)} attempt={attempt_index}\n")
                        except Exception:
                            pass
                        # quick validation
                        if text and ("===JSON===" in text or re.search(r"\{[\s\S]*\}", text)):
                            return text
                    except Exception as exc:
                        attempts.append((name, f"exc:{type(exc).__name__}"))
                        try:
                            os.makedirs(logs_dir, exist_ok=True)
                            with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                                fh.write(f"Gemini attempt {name} failed: {type(exc).__name__}: {exc}\n")
                        except Exception:
                            pass
                # small backoff between outer attempts
                time.sleep(min(1 * (attempt_index + 1), 4))

            try:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"Gemini all attempts failed. genai_dir={genai_dir} attempts={attempts}\n")
            except Exception:
                pass
            return None
        except Exception:
            return None

    # Try OpenAI first
    if provider == "openai" and openai_key:
        try:
            import openai

            openai.api_key = openai_key
            messages = [
                {"role": "system", "content": "You are a helpful research summarization assistant."},
                {"role": "user", "content": full_prompt},
            ]
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1200,
            )
            try:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"OpenAI call model={model} resp_keys={list(resp.keys())}\n")
            except Exception:
                pass
            text = resp["choices"][0]["message"]["content"].strip()
            md, parsed = _extract_json(text)
            # if parsing failed, try one follow-up asking for JSON only
            if parsed and not parsed.get("error"):
                return text
            follow = _request_json_only_openai(messages)
            if follow:
                # combine markdown from earlier with follow-up JSON if possible
                _, parsed2 = _extract_json(follow)
                if parsed2 and not parsed2.get("error"):
                    # ensure we return markdown + marker + json
                    return md + "\n\n===JSON===\n" + json.dumps(parsed2)
            return text
        except Exception as exc:
            try:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"OpenAI call failed: {type(exc).__name__}: {exc}\n")
            except Exception:
                pass

    # Try Gemini (google generative ai) if configured
    if provider == "gemini" and gemini_key:
        try:
            # prefer the new package name if available
            try:
                import google.genai as genai
            except Exception:
                import google.generativeai as genai

            # try to use the high-level Client API when present
            chosen_model = model
            if not chosen_model.startswith("models/"):
                chosen_model = f"models/{chosen_model}" if not chosen_model.startswith("gemini") else f"models/{chosen_model}"
            try:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"Gemini call model={chosen_model}\n")
            except Exception:
                pass

            genai_dir = dir(genai)
            text = ""
            # If the Client class is available we should instantiate with the API key
            ClientClass = getattr(genai, 'Client', None) or getattr(getattr(genai, 'client', {}), 'Client', None)
            if ClientClass:
                try:
                    client = ClientClass(api_key=gemini_key)
                except Exception:
                    os.environ.setdefault('GEMINI_API_KEY', gemini_key)
                    client = ClientClass()
                try:
                    if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
                        resp = client.models.generate_content(model=chosen_model, contents=full_prompt)
                        text = getattr(resp, 'text', str(resp))
                        try:
                            os.makedirs(logs_dir, exist_ok=True)
                            with open(os.path.join(logs_dir, "llm_provider_trace.log"), "a", encoding="utf-8") as fh:
                                fh.write(f"Gemini used client.models.generate_content resp_type={type(resp)}\n")
                        except Exception:
                            pass
                    try:
                        client.close()
                    except Exception:
                        pass
                except Exception as exc:
                    try:
                        with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                            fh.write(f"Gemini client.models.generate_content failed: {type(exc).__name__}: {exc}\n")
                    except Exception:
                        pass

            # If we didn't get text yet, fall back to the more defensive follow-up path
            md, parsed = _extract_json(text)
            if parsed and not parsed.get("error"):
                return text
            follow = _request_json_only_gemini(full_prompt)
            if follow:
                _, parsed2 = _extract_json(follow)
                if parsed2 and not parsed2.get("error"):
                    return md + "\n\n===JSON===\n" + json.dumps(parsed2)
            return text
        except Exception as exc:
            try:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "llm_provider_errors.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"Gemini call failed: {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
            # fall through to fallback
            pass

    # Fallback: local deterministic summary
    bullets = []
    if documents:
        titles = [getattr(d, "title", "") for d in documents[:6]]
        bullets.append(f"Research objective: {objective}")
        bullets.append(f"Retrieved {len(documents)} source documents from sources: {', '.join(sorted({getattr(d, 'source', 'unknown') for d in documents}))}.")
        bullets.append("Major themes to investigate: - transformer-based fusion; - temporal modeling; - modality alignment; - robustness and dataset bias.")
        bullets.append("Representative sources: " + ", ".join(titles))
    else:
        bullets.append("No documents retrieved.")

    md = "\n".join(bullets)
    # include a trivial JSON claim-checks placeholder
    placeholder = {
        "claim_checks": [
            {"claim": "Summary produced by local fallback", "supported": False, "evidence": [], "rationale": "No LLM provider configured"}
        ]
    }
    return md + "\n\n===JSON===\n" + json.dumps(placeholder)


def try_parse_json_section(text: str) -> tuple[str, dict]:
    """If text contains ===JSON=== marker, split and parse the JSON part."""
    marker = "===JSON==="
    if marker in text:
        parts = text.split(marker, 1)
        md = parts[0].strip()
        json_text = parts[1].strip()
        try:
            data = json.loads(json_text)
        except Exception:
            data = {"error": "failed_to_parse_json", "raw": json_text}
        return md, data
    return text, {}
