"""Shared test setup.

Environment variables are set here, at import time and before any ``app`` module is
imported, because ``app.core.config.settings`` is an ``lru_cache``d singleton built
during import. Setting them inside a fixture would be too late.

The default provider is ``mock`` so the suite never makes a network call to an LLM;
tests that exercise a provider path patch it explicitly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="research-agent-tests-"))

# Point ENV_FILE at a path that does not exist so app.core.config skips loading the
# developer's real backend/.env. Without this, live API keys leak into the suite and
# tests that assert on the unconfigured path pass or fail depending on whose machine
# they run on.
os.environ["ENV_FILE"] = str(_TMP / "no-such.env")
os.environ["DEFAULT_LLM_PROVIDER"] = "mock"
os.environ["DEFAULT_LLM_MODEL"] = "gemini-2.0-flash"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ["DATA_DIRECTORY"] = str(_TMP / "data")
os.environ["REPORTS_DIRECTORY"] = str(_TMP / "reports")
os.environ["CHROMA_PERSIST_DIRECTORY"] = str(_TMP / "chroma")
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

import pytest  # noqa: E402

from app.models.schemas import ResearchPlan, SourceDocument  # noqa: E402


@pytest.fixture
def tmp_root() -> Path:
    return _TMP


@pytest.fixture
def documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            title="Multimodal Fusion for Deepfake Detection",
            url="https://arxiv.org/abs/0001",
            authors=["A Researcher"],
            year=2024,
            abstract="We study audio-visual fusion for detecting manipulated media.",
            content="We study audio-visual fusion for detecting manipulated media.",
            source="arxiv",
        ),
        SourceDocument(
            title="Temporal Consistency Cues in Video Forensics",
            url="https://example.org/paper/2",
            authors=["B Scholar"],
            year=2023,
            abstract="Frame-level temporal inconsistency is a reliable forgery signal.",
            content="Frame-level temporal inconsistency is a reliable forgery signal.",
            source="semantic_scholar",
        ),
    ]


@pytest.fixture
def plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Find recent advances in multimodal deepfake detection",
        steps=["Search", "Summarize"],
        search_queries=["multimodal deepfake detection"],
    )
