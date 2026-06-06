from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=6, ge=1, le=20)
    model: str | None = Field(default=None)


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
    preferred_sources: list[str] = Field(default_factory=lambda: ["arxiv", "semantic_scholar", "web"])


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    evidence: list[str] = Field(default_factory=list)
    rationale: str


class ReportPayload(BaseModel):
    markdown: str
    json_summary: dict[str, Any]
    pdf_path: str | None = None


class ResearchRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    query: str
    status: Literal["queued", "planning", "retrieving", "summarizing", "criticizing", "reporting", "complete", "failed"] = "queued"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    plan: ResearchPlan | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    summary: str | None = None
    critic_score: float | None = None
    report: ReportPayload | None = None
    error: str | None = None


class QueryResponse(BaseModel):
    run_id: UUID
    status: str
    plan: ResearchPlan | None = None
    summary: str | None = None
    critic_score: float | None = None
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
