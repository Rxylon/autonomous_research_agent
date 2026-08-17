from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "Multi-agent research pipeline: plan → retrieve → summarize → verify → report. "
        "Call GET /health to see which LLM provider and embedding backend are actually live."
    ),
)

# Note: CORS does not apply to WebSocket handshakes — browsers do not send an
# Origin preflight for them, and Starlette's CORSMiddleware does not filter them.
# The /ws/research endpoint is therefore reachable from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }
