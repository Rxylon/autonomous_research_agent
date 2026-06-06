from __future__ import annotations

import asyncio
from typing import TypedDict

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.report import ReportAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.summarizer import SummarizerAgent
from app.models.schemas import ClaimCheck, ResearchPlan, ReportPayload, SourceDocument


class ResearchState(TypedDict, total=False):
    query: str
    plan: ResearchPlan
    documents: list[SourceDocument]
    summary: str
    critic_score: float
    claim_checks: list[ClaimCheck]
    report: ReportPayload


class ResearchWorkflow:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.retrieval = RetrievalAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.reporter = ReportAgent()

        try:
            from langgraph.graph import END, StateGraph

            self._state_graph = StateGraph(ResearchState)
            self._state_graph.add_node("planner", self._planner_node)
            self._state_graph.add_node("retrieval", self._retrieval_node)
            self._state_graph.add_node("summary", self._summary_node)
            self._state_graph.add_node("critic", self._critic_node)
            self._state_graph.add_node("report", self._report_node)
            self._state_graph.set_entry_point("planner")
            self._state_graph.add_edge("planner", "retrieval")
            self._state_graph.add_edge("retrieval", "summary")
            self._state_graph.add_edge("summary", "critic")
            self._state_graph.add_edge("critic", "report")
            self._state_graph.add_edge("report", END)
            self._compiled = self._state_graph.compile()
        except Exception:
            self._state_graph = None
            self._compiled = None

    def _planner_node(self, state: ResearchState) -> ResearchState:
        return {"plan": self.planner.invoke(state["query"])}

    def _retrieval_node(self, state: ResearchState) -> ResearchState:
        return {"documents": self.retrieval.invoke(state["plan"])}

    def _summary_node(self, state: ResearchState) -> ResearchState:
        return {"summary": self.summarizer.invoke(state["plan"], state["documents"])}

    def _critic_node(self, state: ResearchState) -> ResearchState:
        critic_score, claim_checks = self.critic.invoke(state["summary"], state["documents"])
        return {"critic_score": critic_score, "claim_checks": claim_checks}

    def _report_node(self, state: ResearchState) -> ResearchState:
        return {
            "report": self.reporter.invoke(
                query=state["query"],
                plan=state["plan"],
                summary=state["summary"],
                documents=state["documents"],
                critic_score=state["critic_score"],
                claim_checks=state.get("claim_checks", []),
            )
        }

    async def _emit(self, emitter, payload):
        if emitter is not None:
            await emitter(payload)

    async def run_with_progress(self, query: str, emitter=None) -> ResearchState:
        state: ResearchState = {"query": query}
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "planning",
                "status": "running",
                "message": "Breaking the query into executable research steps.",
                "details": {"query": query},
            },
        )
        state.update(self._planner_node(state))
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "planning",
                "status": "complete",
                "message": "Planner generated a research plan.",
                "details": {"steps": state["plan"].steps, "search_queries": state["plan"].search_queries},
            },
        )

        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "retrieval",
                "status": "running",
                "message": "Searching papers and indexing source material.",
                "details": {"search_queries": state["plan"].search_queries},
            },
        )
        state.update(self._retrieval_node(state))
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "retrieval",
                "status": "complete",
                "message": f'Retrieved {len(state["documents"])} source documents.',
                "details": {"documents": [document.model_dump(mode="json") for document in state["documents"]]},
            },
        )

        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "summarizing",
                "status": "running",
                "message": "Summarizing evidence into major approaches and trends.",
            },
        )
        state.update(self._summary_node(state))
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "summarizing",
                "status": "complete",
                "message": "Summary generated.",
                "details": {"summary": state["summary"]},
            },
        )

        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "critic",
                "status": "running",
                "message": "Checking claims against the retrieved source set.",
            },
        )
        state.update(self._critic_node(state))
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "critic",
                "status": "complete",
                "message": f'Critic score: {state["critic_score"]:.2f}',
                "details": {"critic_score": state["critic_score"], "claim_checks": [check.model_dump(mode="json") for check in state.get("claim_checks", [])]},
            },
        )

        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "report",
                "status": "running",
                "message": "Generating Markdown, JSON, and PDF report artifacts.",
            },
        )
        state.update(self._report_node(state))
        await self._emit(
            emitter,
            {
                "type": "progress",
                "stage": "report",
                "status": "complete",
                "message": "Report generated.",
                "details": {"pdf_path": state["report"].pdf_path},
            },
        )
        return state

    def run(self, query: str) -> ResearchState:
        return asyncio.run(self.run_with_progress(query))
