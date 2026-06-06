from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect

from app.graph.workflow import ResearchWorkflow
from app.models.schemas import QueryRequest, QueryResponse, ResearchProgressEvent, ResearchRun, SourceDocument, UploadResponse
from app.services.history_store import HistoryStore
from app.services.vector_store import get_vector_store

router = APIRouter()
workflow = ResearchWorkflow()
history_store = HistoryStore()
vector_store = get_vector_store()


@router.post("/query", response_model=QueryResponse)
def query_research(request: QueryRequest) -> QueryResponse:
    run = ResearchRun(query=request.query, status="planning")
    try:
        state = workflow.run(request.query)
        run.status = "complete"
        run.plan = state.get("plan")
        run.sources = state.get("documents", [])
        run.summary = state.get("summary")
        run.critic_score = state.get("critic_score")
        run.report = state.get("report")
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
    finally:
        history_store.append(run)

    return QueryResponse(
        run_id=run.id,
        status=run.status,
        plan=run.plan,
        summary=run.summary,
        critic_score=run.critic_score,
        report_markdown=run.report.markdown if run.report else None,
        sources=run.sources,
        error=run.error,
    )


@router.websocket("/ws/research")
async def research_progress_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        request = QueryRequest.model_validate(payload)
        await websocket.send_json(
            ResearchProgressEvent(
                type="progress",
                stage="queued",
                status="accepted",
                message="Research request accepted.",
                details={"query": request.query},
            ).model_dump(mode="json")
        )

        async def emit(event: dict) -> None:
            await websocket.send_json(event)

        state = await workflow.run_with_progress(request.query, emitter=emit)
        run = ResearchRun(
            query=request.query,
            status="complete",
            plan=state.get("plan"),
            sources=state.get("documents", []),
            summary=state.get("summary"),
            critic_score=state.get("critic_score"),
            report=state.get("report"),
        )
        history_store.append(run)
        await websocket.send_json(
            ResearchProgressEvent(
                type="result",
                stage="complete",
                status="complete",
                message="Research run completed.",
                result=QueryResponse(
                    run_id=run.id,
                    status=run.status,
                    plan=run.plan,
                    summary=run.summary,
                    critic_score=run.critic_score,
                    report_markdown=run.report.markdown if run.report else None,
                    sources=run.sources,
                    error=run.error,
                ).model_dump(mode="json"),
            ).model_dump(mode="json")
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json(
                ResearchProgressEvent(
                    type="error",
                    stage="error",
                    status="failed",
                    message=str(exc),
                ).model_dump(mode="json")
            )
        finally:
            await websocket.close()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    chunks_indexed = vector_store.add_text(
        text,
        document=SourceDocument(
            title=file.filename or "uploaded-document",
            url=None,
            authors=[],
            year=None,
            abstract=None,
            content=text,
            source="upload",
        ),
    )
    return UploadResponse(
        document_id=file.filename or "document",
        filename=file.filename or "document",
        chunks_indexed=chunks_indexed,
        metadata={"content_length": len(text)},
    )


@router.get("/history")
def get_history() -> list[ResearchRun]:
    return history_store.list(limit=25)


@router.get("/history/{run_id}")
def get_history_item(run_id: str) -> ResearchRun | None:
    from uuid import UUID

    return history_store.get(UUID(run_id))
