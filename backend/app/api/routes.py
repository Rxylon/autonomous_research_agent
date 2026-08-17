from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.agents.retrieval import extract_pdf_text_from_bytes
from app.core.config import settings
from app.graph.workflow import ResearchWorkflow
from app.models.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ResearchProgressEvent,
    ResearchRun,
    SourceDocument,
    UploadResponse,
    utc_now,
)
from app.services.embeddings import embedding_backend
from app.services.history_store import HistoryStore
from app.services.llm_provider import last_provider_error, resolve_provider_config
from app.services.vector_store import get_vector_store

router = APIRouter()
workflow = ResearchWorkflow()
history_store = HistoryStore()
vector_store = get_vector_store()

# Guard against a multi-hundred-MB upload exhausting memory on a free-tier host,
# where the whole file is read before it is parsed.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _run_from_state(query: str, state: dict) -> ResearchRun:
    return ResearchRun(
        query=query,
        status="complete",
        plan=state.get("plan"),
        sources=state.get("documents", []),
        summary=state.get("summary"),
        critic_score=state.get("critic_score"),
        critic_method=state.get("critic_method"),
        claim_checks=state.get("claim_checks", []),
        report=state.get("report"),
        updated_at=utc_now(),
    )


def _response_from_run(run: ResearchRun) -> QueryResponse:
    return QueryResponse(
        run_id=run.id,
        status=run.status,
        plan=run.plan,
        summary=run.summary,
        critic_score=run.critic_score,
        critic_method=run.critic_method,
        claim_checks=run.claim_checks,
        report_markdown=run.report.markdown if run.report else None,
        sources=run.sources,
        error=run.error,
    )


@router.post("/query", response_model=QueryResponse)
async def query_research(request: QueryRequest) -> QueryResponse:
    """Run the full pipeline and return the finished result.

    A failed run is still persisted to history with ``status="failed"`` and the
    error text, so a run never disappears silently.
    """
    try:
        state = await workflow.run_with_progress(request.query, top_k=request.top_k, model=request.model)
        run = _run_from_state(request.query, state)
    except Exception as exc:
        run = ResearchRun(query=request.query, status="failed", error=f"{type(exc).__name__}: {exc}")

    history_store.append(run)
    return _response_from_run(run)


@router.websocket("/ws/research")
async def research_progress_stream(websocket: WebSocket) -> None:
    """Stream stage-by-stage progress, then the final result.

    Every blocking stage runs in a worker thread (see ``app.graph.workflow``), so
    this handler stays responsive and concurrent runs do not serialise behind
    each other.
    """
    await websocket.accept()
    query = ""
    try:
        request = QueryRequest.model_validate(await websocket.receive_json())
        query = request.query

        await websocket.send_json(
            ResearchProgressEvent(
                type="progress",
                stage="queued",
                status="accepted",
                message="Research request accepted.",
                details={"query": request.query, "engine": workflow.engine},
            ).model_dump(mode="json")
        )

        async def emit(event: dict) -> None:
            await websocket.send_json(event)

        state = await workflow.run_with_progress(
            request.query, emitter=emit, top_k=request.top_k, model=request.model
        )
        run = _run_from_state(request.query, state)
        history_store.append(run)

        await websocket.send_json(
            ResearchProgressEvent(
                type="result",
                stage="complete",
                status="complete",
                message="Research run completed.",
                result=_response_from_run(run).model_dump(mode="json"),
            ).model_dump(mode="json")
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        # Persist the failure too, so a run that died mid-stream is still auditable
        # in history rather than vanishing.
        if query:
            history_store.append(
                ResearchRun(query=query, status="failed", error=f"{type(exc).__name__}: {exc}")
            )
        try:
            await websocket.send_json(
                ResearchProgressEvent(
                    type="error",
                    stage="error",
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                ).model_dump(mode="json")
            )
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Index a PDF or text file into the vector store.

    PDFs are parsed with PyMuPDF. Decoding PDF bytes as UTF-8 — which this route
    used to do — yields binary noise, so uploads appeared to succeed while
    indexing nothing searchable.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(payload) // 1024 // 1024} MB; the limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    filename = file.filename or "uploaded-document"
    is_pdf = payload[:5] == b"%PDF-" or filename.lower().endswith(".pdf")

    if is_pdf:
        try:
            text = extract_pdf_text_from_bytes(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not extract text from PDF: {exc}") from exc
        content_type = "pdf"
    else:
        text = payload.decode("utf-8", errors="replace")
        content_type = "text"

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. Scanned PDFs need OCR, which this service does not perform.",
        )

    document = SourceDocument(
        title=filename,
        url=None,
        authors=[],
        year=None,
        abstract=None,
        content=text,
        source="upload",
    )
    chunks_indexed = vector_store.add_text(text, document=document)

    return UploadResponse(
        document_id=filename,
        filename=filename,
        chunks_indexed=chunks_indexed,
        metadata={
            "content_length": len(text),
            "content_type": content_type,
            "bytes_received": len(payload),
            "embedding_backend": embedding_backend(),
        },
    )


@router.get("/history", response_model=list[ResearchRun])
def get_history(limit: int = Query(default=25, ge=1, le=200)) -> list[ResearchRun]:
    return history_store.list(limit=limit)


def _load_run(run_id: str) -> ResearchRun:
    try:
        parsed = UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{run_id!r} is not a valid UUID.") from exc

    run = history_store.get(parsed)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No research run with id {run_id}.")
    return run


@router.get("/history/{run_id}", response_model=ResearchRun)
def get_history_item(run_id: str) -> ResearchRun:
    return _load_run(run_id)


@router.get("/reports/{run_id}.md", response_class=PlainTextResponse)
def download_report_markdown(run_id: str) -> PlainTextResponse:
    run = _load_run(run_id)
    if run.report is None:
        raise HTTPException(status_code=404, detail="This run has no report.")
    return PlainTextResponse(
        run.report.markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="report-{run_id}.md"'},
    )


@router.get("/reports/{run_id}.json")
def download_report_json(run_id: str) -> Response:
    run = _load_run(run_id)
    if run.report is None:
        raise HTTPException(status_code=404, detail="This run has no report.")
    return Response(
        json.dumps(run.report.json_summary, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report-{run_id}.json"'},
    )


@router.get("/reports/{run_id}.pdf")
def download_report_pdf(run_id: str) -> FileResponse:
    """Serve the generated PDF.

    Unlike the Markdown and JSON bodies, the PDF only exists on disk. Hosts with an
    ephemeral filesystem lose it on restart, which surfaces here as a 410 rather
    than a misleading 404.
    """
    run = _load_run(run_id)
    if run.report is None or not run.report.pdf_path:
        raise HTTPException(status_code=404, detail="No PDF was generated for this run.")

    path = Path(run.report.pdf_path)
    if not path.is_file():
        raise HTTPException(
            status_code=410,
            detail=(
                "The PDF for this run is no longer on disk. Report files are stored on the "
                "local filesystem, which is ephemeral on free hosting tiers. Use the .md or "
                ".json download, which are served from run history."
            ),
        )
    return FileResponse(path, media_type="application/pdf", filename=f"report-{run_id}.pdf")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report what is actually configured, not just that the process is alive.

    ``llm_configured=false`` is the single most useful signal here: it means every
    summary will come from the local deterministic fallback.
    """
    config = resolve_provider_config()
    return HealthResponse(
        status="ok",
        llm_provider=config.provider,
        llm_configured=config.usable,
        llm_model=config.model,
        embedding_backend=embedding_backend(),
        vector_store_documents=vector_store.count(),
        last_llm_error=last_provider_error(),
    )


@router.get("/config")
def runtime_config() -> dict:
    """Non-secret runtime configuration, for debugging a deployment."""
    config = resolve_provider_config()
    return {
        "environment": settings.environment,
        "engine": workflow.engine,
        "llm_provider": config.provider,
        "llm_model": config.model,
        "llm_configured": config.usable,
        "embedding_backend": embedding_backend(),
        "embedding_model": settings.embedding_model,
        "vector_store_collection": vector_store.collection_name,
        "vector_store_documents": vector_store.count(),
        "cors_origins": settings.cors_origins,
        "reports_directory": settings.reports_directory,
        "data_directory": settings.data_directory,
    }
