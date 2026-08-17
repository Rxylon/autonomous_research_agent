"""Tests that the pipeline actually runs through the compiled LangGraph."""

from __future__ import annotations

import asyncio

import pytest

from app.graph import workflow as workflow_module
from app.graph.workflow import STAGES, ResearchWorkflow
from app.models.schemas import RetrievalResult


@pytest.fixture
def offline_workflow(monkeypatch, documents) -> ResearchWorkflow:
    """A workflow whose retrieval is stubbed, so no test touches the network."""
    flow = ResearchWorkflow()
    monkeypatch.setattr(
        flow.retrieval,
        "invoke",
        lambda plan, top_k=5: RetrievalResult(
            documents=documents,
            searched_count=len(documents),
            recalled_count=1,
            chunks_indexed=4,
            vector_store_size=4,
        ),
    )
    return flow


class TestGraphIsTheExecutionPath:
    """Regression: the graph was compiled in `__init__` and then never used —
    `run_with_progress` called each node function directly, so the declared edges
    had no effect on execution.
    """

    def test_langgraph_compiles(self):
        assert ResearchWorkflow().engine == "langgraph", (
            "langgraph failed to compile; the run would silently fall back to the sequential path"
        )

    def test_compiled_graph_is_driven_not_bypassed(self, offline_workflow, monkeypatch):
        flow = offline_workflow
        assert flow._compiled is not None

        astream_used = {"value": False}
        original = flow._compiled.astream

        def tracking_astream(*args, **kwargs):
            astream_used["value"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(flow._compiled, "astream", tracking_astream)
        asyncio.run(flow.run_with_progress("multimodal deepfake detection"))
        assert astream_used["value"], "run_with_progress bypassed the compiled graph"

    def test_sequential_fallback_produces_the_same_keys(self, offline_workflow):
        flow = offline_workflow
        graph_state = asyncio.run(flow.run_with_progress("multimodal deepfake detection"))

        flow._compiled = None
        assert flow.engine == "sequential"
        sequential_state = asyncio.run(flow.run_with_progress("multimodal deepfake detection"))

        assert set(graph_state) == set(sequential_state)


class TestProgressStream:
    def test_every_stage_reports_running_then_complete(self, offline_workflow):
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        asyncio.run(offline_workflow.run_with_progress("multimodal deepfake detection", emitter=emit))

        for stage in STAGES:
            statuses = [e["status"] for e in events if e["stage"] == stage]
            assert statuses == ["running", "complete"], f"stage {stage} reported {statuses}"

    def test_stages_arrive_in_graph_order(self, offline_workflow):
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        asyncio.run(offline_workflow.run_with_progress("query about detection", emitter=emit))
        order = [e["stage"] for e in events if e["status"] == "complete"]
        assert order == list(STAGES)

    def test_retrieval_event_reports_vector_store_recall(self, offline_workflow):
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        asyncio.run(offline_workflow.run_with_progress("query about detection", emitter=emit))
        retrieval = next(e for e in events if e["stage"] == "retrieval" and e["status"] == "complete")
        assert retrieval["details"]["recalled_count"] == 1
        assert "recalled from the vector store" in retrieval["message"]

    def test_critic_event_flags_a_heuristic_score(self, offline_workflow):
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        asyncio.run(offline_workflow.run_with_progress("query about detection", emitter=emit))
        critic = next(e for e in events if e["stage"] == "critic" and e["status"] == "complete")
        assert critic["details"]["critic_method"] == "heuristic"
        assert "not model-verified" in critic["message"]


class TestStateHygiene:
    def test_emitter_is_stripped_from_the_returned_state(self, offline_workflow):
        async def emit(event: dict) -> None:
            return None

        state = asyncio.run(offline_workflow.run_with_progress("query about detection", emitter=emit))
        assert "emitter" not in state, "a live callback would be serialised into the response body"

    def test_run_produces_every_stage_output(self, offline_workflow):
        state = asyncio.run(offline_workflow.run_with_progress("query about detection"))
        assert state["plan"].objective == "query about detection"
        assert state["documents"]
        assert state["summary"]
        assert state["critic_score"] is not None
        assert state["report"].markdown.startswith("# Research Report")

    def test_blocking_work_is_offloaded_from_the_event_loop(self, offline_workflow, monkeypatch):
        """Regression: agent calls ran inline on the event loop, so one run froze
        every other WebSocket connection for its duration.
        """
        threaded = {"count": 0}
        original = workflow_module.asyncio.to_thread

        async def counting_to_thread(func, *args, **kwargs):
            threaded["count"] += 1
            return await original(func, *args, **kwargs)

        monkeypatch.setattr(workflow_module.asyncio, "to_thread", counting_to_thread)
        asyncio.run(offline_workflow.run_with_progress("query about detection"))
        assert threaded["count"] == len(STAGES), (
            f"expected all {len(STAGES)} stages off the loop, saw {threaded['count']}"
        )

    def test_sync_run_wrapper_works(self, offline_workflow):
        state = offline_workflow.run("query about detection")
        assert state["report"].markdown
