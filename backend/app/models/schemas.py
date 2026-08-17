from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp.

    ``datetime.utcnow()`` is deprecated from Python 3.12 and returns a naive value,
    which then compares unequal to any aware timestamp read back from history.
    """
    return datetime.now(timezone.utc)


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20, description="Results requested per search query.")
    model: str | None = Field(default=None, description="Override DEFAULT_LLM_MODEL for this run.")


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    title: str
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    content: str = ""
    source: str = "unknown"


class ResearchPlan(BaseModel):
    objective: str
    steps: list[str]
    search_queries: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=lambda: ["arxiv", "semantic_scholar", "crossref"])


class RetrievalResult(BaseModel):
    """Documents plus provenance for how they were obtained."""

    documents: list[SourceDocument] = Field(default_factory=list)
    searched_count: int = 0
    recalled_count: int = Field(default=0, description="Documents recalled from the vector store, not live search.")
    chunks_indexed: int = 0
    vector_store_size: int = 0


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class CriticVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    method: Literal["llm", "heuristic", "empty"] = Field(
        default="empty",
        description="How the score was produced. `heuristic` means no LLM verified these claims.",
    )


class ReportPayload(BaseModel):
    markdown: str
    json_summary: dict[str, Any]
    pdf_path: str | None = None


class ResearchRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    query: str
    status: Literal[
        "queued", "planning", "retrieving", "summarizing", "criticizing", "reporting", "complete", "failed"
    ] = "queued"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    plan: ResearchPlan | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    summary: str | None = None
    critic_score: float | None = None
    critic_method: str | None = None
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    report: ReportPayload | None = None
    error: str | None = None


class QueryResponse(BaseModel):
    run_id: UUID
    status: str
    plan: ResearchPlan | None = None
    summary: str | None = None
    critic_score: float | None = None
    critic_method: str | None = None
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    report_markdown: str | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    error: str | None = None


class ResearchProgressEvent(BaseModel):
    type: Literal["progress", "result", "error"]
    stage: str
    status: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_configured: bool = Field(description="True when the selected provider has an API key available.")
    llm_model: str
    embedding_backend: str
    vector_store_documents: int
    last_llm_error: str | None = Field(
        default=None,
        description=(
            "Most recent provider failure, if any. A key can be present while every call "
            "fails (invalid key, quota, rate limit) — in that state summaries silently come "
            "from the local fallback, and this field is how you find out."
        ),
    )
