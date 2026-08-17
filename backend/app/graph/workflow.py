"""LangGraph orchestration for the five-stage research pipeline.

Two things this module is careful about:

**The compiled graph is what actually runs.** ``run_with_progress`` drives
``self._compiled.astream(...)``, so the edges declared below determine execution
order. An earlier version compiled the graph and then called the node functions by
hand, which meant the graph was decoration. If LangGraph is unavailable the
sequential fallback in :meth:`_run_sequential` takes over, and
:attr:`ResearchWorkflow.engine` reports which path is live.

**Nothing blocking runs on the event loop.** Every node is ``async`` but the agent
work inside it — HTTP paper search, LLM calls, embedding, PDF writing — is
synchronous and slow, so each is dispatched with ``asyncio.to_thread``. Without
that, one research run would freeze every other WebSocket connection on the worker
for the duration.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypedDict

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.report import ReportAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.summarizer import SummarizerAgent
from app.models.schemas import (
    ClaimCheck,
    CriticVerdict,
    ReportPayload,
    ResearchPlan,
    RetrievalResult,
    SourceDocument,
)

Emitter = Callable[[dict[str, Any]], Awaitable[None]]

STAGES = ("planning", "retrieval", "summarizing", "critic", "report")


class ResearchState(TypedDict, total=False):
    # Inputs
    query: str
    top_k: int
    model: str | None
    # Progress callback. Carried through the graph so nodes can report as they run;
    # stripped before the state is persisted or returned to a client.
    emitter: Any
    # Stage outputs
    plan: ResearchPlan
    documents: list[SourceDocument]
    retrieval: RetrievalResult
    summary: str
    critic_score: float
    critic_method: str
    claim_checks: list[ClaimCheck]
    report: ReportPayload


async def _emit(state: ResearchState, payload: dict[str, Any]) -> None:
    emitter = state.get("emitter")
    if emitter is None:
        return
    await emitter(payload)


async def _progress(state: ResearchState, stage: str, status: str, message: str, details: dict | None = None) -> None:
    await _emit(
        state,
        {
            "type": "progress",
            "stage": stage,
            "status": status,
            "message": message,
            "details": details or {},
        },
    )


class ResearchWorkflow:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.retrieval = RetrievalAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.reporter = ReportAgent()

        self._state_graph = None
        self._compiled = None
        self._graph_error: str | None = None

        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(ResearchState)
            graph.add_node("planner", self._planner_node)
            graph.add_node("retrieval", self._retrieval_node)
            graph.add_node("summary", self._summary_node)
            graph.add_node("critic", self._critic_node)
            graph.add_node("report", self._report_node)
            graph.set_entry_point("planner")
            graph.add_edge("planner", "retrieval")
            graph.add_edge("retrieval", "summary")
            graph.add_edge("summary", "critic")
            graph.add_edge("critic", "report")
            graph.add_edge("report", END)

            self._state_graph = graph
            self._compiled = graph.compile()
        except Exception as exc:  # pragma: no cover - exercised only without langgraph
            self._graph_error = f"{type(exc).__name__}: {exc}"

    @property
    def engine(self) -> str:
        """``langgraph`` when the compiled graph drives the run, else ``sequential``."""
        return "langgraph" if self._compiled is not None else "sequential"

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    async def _planner_node(self, state: ResearchState) -> ResearchState:
        query = state["query"]
        await _progress(state, "planning", "running", "Breaking the query into executable research steps.", {"query": query})
        plan = await asyncio.to_thread(self.planner.invoke, query)
        await _progress(
            state,
            "planning",
            "complete",
            "Planner generated a research plan.",
            {"steps": plan.steps, "search_queries": plan.search_queries},
        )
        return {"plan": plan}

    async def _retrieval_node(self, state: ResearchState) -> ResearchState:
        plan = state["plan"]
        await _progress(
            state,
            "retrieval",
            "running",
            "Searching papers and indexing source material.",
            {"search_queries": plan.search_queries},
        )
        result: RetrievalResult = await asyncio.to_thread(
            self.retrieval.invoke, plan, state.get("top_k", 5)
        )
        recalled = result.recalled_count
        message = f"Retrieved {len(result.documents)} source documents."
        if recalled:
            message += f" {recalled} recalled from the vector store (uploads or earlier runs)."
        await _progress(
            state,
            "retrieval",
            "complete",
            message,
            {
                "documents": [document.model_dump(mode="json") for document in result.documents],
                "searched_count": result.searched_count,
                "recalled_count": recalled,
                "chunks_indexed": result.chunks_indexed,
                "vector_store_documents": result.vector_store_size,
            },
        )
        return {"documents": result.documents, "retrieval": result}

    async def _summary_node(self, state: ResearchState) -> ResearchState:
        await _progress(state, "summarizing", "running", "Summarizing evidence into major approaches and trends.")
        summary = await asyncio.to_thread(
            self.summarizer.invoke, state["plan"], state["documents"], state.get("model")
        )
        await _progress(state, "summarizing", "complete", "Summary generated.", {"summary": summary})
        return {"summary": summary}

    async def _critic_node(self, state: ResearchState) -> ResearchState:
        await _progress(state, "critic", "running", "Checking the summary's claims against the retrieved sources.")
        verdict: CriticVerdict = await asyncio.to_thread(
            self.critic.invoke, state["summary"], state["documents"], state.get("model")
        )
        qualifier = " (keyword heuristic, not model-verified)" if verdict.method == "heuristic" else ""
        await _progress(
            state,
            "critic",
            "complete",
            f"Critic score: {verdict.score:.2f}{qualifier}",
            {
                "critic_score": verdict.score,
                "critic_method": verdict.method,
                "claim_checks": [check.model_dump(mode="json") for check in verdict.claim_checks],
            },
        )
        return {
            "critic_score": verdict.score,
            "critic_method": verdict.method,
            "claim_checks": verdict.claim_checks,
        }

    async def _report_node(self, state: ResearchState) -> ResearchState:
        await _progress(state, "report", "running", "Generating Markdown, JSON, and PDF report artifacts.")
        report = await asyncio.to_thread(
            self.reporter.invoke,
            state["query"],
            state["plan"],
            state["summary"],
            state["documents"],
            state.get("critic_score", 0.0),
            state.get("claim_checks", []),
            state.get("critic_method", "empty"),
        )
        await _progress(
            state,
            "report",
            "complete",
            "Report generated." if report.pdf_path else "Report generated (PDF unavailable on this host).",
            {"pdf_path": report.pdf_path},
        )
        return {"report": report}

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def _run_sequential(self, state: ResearchState) -> ResearchState:
        """Fallback path used only when LangGraph could not be compiled.

        Mirrors the edges declared in ``__init__``; if you add a node there, add it here.
        """
        for node in (
            self._planner_node,
            self._retrieval_node,
            self._summary_node,
            self._critic_node,
            self._report_node,
        ):
            state.update(await node(state))
        return state

    async def run_with_progress(
        self,
        query: str,
        emitter: Emitter | None = None,
        top_k: int = 5,
        model: str | None = None,
    ) -> ResearchState:
        state: ResearchState = {"query": query, "top_k": top_k, "model": model, "emitter": emitter}

        if self._compiled is None:
            final = await self._run_sequential(state)
        else:
            # stream_mode="values" yields the accumulated state after each superstep,
            # so the last item is the completed run.
            final = state
            async for snapshot in self._compiled.astream(state, stream_mode="values"):
                final = snapshot

        # The emitter is a live callback, not research output — never let it reach a
        # response body or the history file.
        return {key: value for key, value in final.items() if key != "emitter"}

    def run(self, query: str, top_k: int = 5, model: str | None = None) -> ResearchState:
        """Blocking entry point for the synchronous ``POST /query`` route."""
        return asyncio.run(self.run_with_progress(query, top_k=top_k, model=model))
