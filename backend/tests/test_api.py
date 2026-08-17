"""End-to-end API tests via FastAPI's TestClient. No network calls."""

from __future__ import annotations

import json

import pytest

from app.models.schemas import RetrievalResult, SourceDocument

pytest.importorskip("fastapi", reason="API tests need fastapi installed")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import routes  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client(monkeypatch, documents) -> TestClient:
    """A client whose retrieval is stubbed so tests never hit arXiv."""
    monkeypatch.setattr(
        routes.workflow.retrieval,
        "invoke",
        lambda plan, top_k=5: RetrievalResult(
            documents=documents,
            searched_count=len(documents),
            recalled_count=0,
            chunks_indexed=2,
            vector_store_size=2,
        ),
    )
    return TestClient(app)


def _minimal_pdf() -> bytes:
    """Build a one-page PDF with real extractable text, using reportlab."""
    import io

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, "Holographic interferometry reveals synthetic media artifacts.")
    pdf.save()
    return buffer.getvalue()


class TestHealthAndConfig:
    def test_health_reports_whether_an_llm_is_actually_configured(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        # conftest sets provider=mock with no keys.
        assert body["llm_configured"] is False
        assert body["llm_provider"] == "mock"
        assert body["embedding_backend"] in {"openai", "sentence-transformers", "hash"}

    def test_config_exposes_the_execution_engine_and_no_secrets(self, client):
        body = client.get("/config").json()
        assert body["engine"] in {"langgraph", "sequential"}
        serialised = json.dumps(body).lower()
        assert "api_key" not in serialised
        assert "sk-" not in serialised

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_health_surfaces_a_provider_failure(self, client, monkeypatch):
        """A present-but-broken key is the silent failure mode: every summary falls
        back to the local path while health still says configured. Health must say so."""
        from app.services import llm_provider

        monkeypatch.setattr(llm_provider, "_last_error", "gemini call failed: 400 API_KEY_INVALID")
        assert "API_KEY_INVALID" in client.get("/health").json()["last_llm_error"]

    def test_health_reports_no_error_when_nothing_has_failed(self, client, monkeypatch):
        from app.services import llm_provider

        monkeypatch.setattr(llm_provider, "_last_error", None)
        assert client.get("/health").json()["last_llm_error"] is None


class TestQueryEndpoint:
    def test_full_run_returns_a_complete_result(self, client):
        response = client.post("/query", json={"query": "multimodal deepfake detection"})
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "complete"
        assert body["plan"]["objective"] == "multimodal deepfake detection"
        assert body["sources"]
        assert body["summary"]
        assert body["report_markdown"].startswith("# Research Report")

    def test_response_labels_how_the_critic_scored(self, client):
        body = client.post("/query", json={"query": "multimodal deepfake detection"}).json()
        assert body["critic_method"] in {"llm", "heuristic", "empty"}
        assert 0.0 <= body["critic_score"] <= 1.0

    def test_short_query_is_rejected(self, client):
        assert client.post("/query", json={"query": "ai"}).status_code == 422

    def test_out_of_range_top_k_is_rejected(self, client):
        assert client.post("/query", json={"query": "valid query", "top_k": 99}).status_code == 422

    def test_a_failing_run_is_recorded_rather_than_raising(self, client, monkeypatch):
        monkeypatch.setattr(
            routes.workflow.retrieval,
            "invoke",
            lambda plan, top_k=5: (_ for _ in ()).throw(RuntimeError("upstream exploded")),
        )
        body = client.post("/query", json={"query": "a query that will fail"}).json()
        assert body["status"] == "failed"
        assert "upstream exploded" in body["error"]

        history = client.get("/history").json()
        assert any(run["query"] == "a query that will fail" and run["status"] == "failed" for run in history)


class TestWebSocket:
    def test_stream_emits_every_stage_then_a_result(self, client):
        with client.websocket_connect("/ws/research") as socket:
            socket.send_json({"query": "multimodal deepfake detection"})
            messages = []
            while True:
                message = socket.receive_json()
                messages.append(message)
                if message["type"] in {"result", "error"}:
                    break

        assert messages[-1]["type"] == "result", messages[-1]
        stages = [m["stage"] for m in messages if m["type"] == "progress" and m["status"] == "complete"]
        assert stages == ["planning", "retrieval", "summarizing", "critic", "report"]
        assert messages[-1]["result"]["report_markdown"]

    def test_stream_reports_errors_and_records_the_failed_run(self, client, monkeypatch):
        monkeypatch.setattr(
            routes.workflow.retrieval,
            "invoke",
            lambda plan, top_k=5: (_ for _ in ()).throw(RuntimeError("stream failure")),
        )
        with client.websocket_connect("/ws/research") as socket:
            socket.send_json({"query": "a streaming query that fails"})
            while True:
                message = socket.receive_json()
                if message["type"] in {"result", "error"}:
                    break

        assert message["type"] == "error"
        assert "stream failure" in message["message"]
        history = client.get("/history").json()
        assert any(run["query"] == "a streaming query that fails" and run["status"] == "failed" for run in history)

    def test_result_payload_never_leaks_the_emitter(self, client):
        with client.websocket_connect("/ws/research") as socket:
            socket.send_json({"query": "multimodal deepfake detection"})
            while True:
                message = socket.receive_json()
                if message["type"] == "result":
                    break
        assert "emitter" not in message["result"]


class TestHistory:
    def test_history_lists_runs_newest_first(self, client):
        client.post("/query", json={"query": "first history query"})
        client.post("/query", json={"query": "second history query"})

        runs = client.get("/history").json()
        assert runs[0]["query"] == "second history query"

    def test_history_respects_limit(self, client):
        client.post("/query", json={"query": "a query for the limit test"})
        assert len(client.get("/history?limit=1").json()) == 1

    def test_single_run_is_retrievable(self, client):
        run_id = client.post("/query", json={"query": "retrievable single run"}).json()["run_id"]
        body = client.get(f"/history/{run_id}").json()
        assert body["id"] == run_id
        assert body["report"]["markdown"]

    def test_malformed_uuid_is_a_400_not_a_500(self, client):
        """Regression: the route called UUID(run_id) unguarded, so any non-UUID
        path segment surfaced as an unhandled 500."""
        response = client.get("/history/not-a-uuid")
        assert response.status_code == 400
        assert "not a valid UUID" in response.json()["detail"]

    def test_unknown_run_is_a_404(self, client):
        response = client.get("/history/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestReportDownloads:
    def test_markdown_download(self, client):
        run_id = client.post("/query", json={"query": "downloadable markdown run"}).json()["run_id"]
        response = client.get(f"/reports/{run_id}.md")
        assert response.status_code == 200
        assert response.text.startswith("# Research Report")
        assert "attachment" in response.headers["content-disposition"]

    def test_json_download_is_valid_json(self, client):
        run_id = client.post("/query", json={"query": "downloadable json run"}).json()["run_id"]
        response = client.get(f"/reports/{run_id}.json")
        assert response.status_code == 200
        payload = json.loads(response.text)
        assert payload["query"] == "downloadable json run"
        assert "claim_checks" in payload

    def test_pdf_download(self, client):
        run_id = client.post("/query", json={"query": "downloadable pdf run"}).json()["run_id"]
        response = client.get(f"/reports/{run_id}.pdf")
        assert response.status_code == 200
        assert response.content[:5] == b"%PDF-"

    def test_missing_pdf_on_disk_is_a_410_with_an_explanation(self, client, monkeypatch):
        run_id = client.post("/query", json={"query": "pdf that gets deleted"}).json()["run_id"]

        from pathlib import Path

        run = routes.history_store.get(__import__("uuid").UUID(run_id))
        Path(run.report.pdf_path).unlink()

        response = client.get(f"/reports/{run_id}.pdf")
        assert response.status_code == 410
        assert "ephemeral" in response.json()["detail"]

    def test_download_of_unknown_run_is_a_404(self, client):
        assert client.get("/reports/00000000-0000-0000-0000-000000000000.md").status_code == 404


class TestUpload:
    def test_pdf_upload_extracts_real_text(self, client):
        """Regression: the route decoded PDF bytes as UTF-8, so a PDF upload
        reported success while indexing binary noise."""
        response = client.post(
            "/upload",
            files={"file": ("paper.pdf", _minimal_pdf(), "application/pdf")},
        )
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["metadata"]["content_type"] == "pdf"
        assert body["chunks_indexed"] >= 1

    def test_uploaded_pdf_becomes_searchable(self, client):
        client.post("/upload", files={"file": ("paper.pdf", _minimal_pdf(), "application/pdf")})
        hits = routes.vector_store.search("holographic interferometry synthetic media", top_k=3)
        assert any("interferometry" in (hit["content"] or "").lower() for hit in hits)

    def test_text_upload_still_works(self, client):
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"Plain text notes about diffusion model watermarks.", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["content_type"] == "text"

    def test_empty_file_is_rejected(self, client):
        response = client.post("/upload", files={"file": ("empty.txt", b"", "text/plain")})
        assert response.status_code == 400

    def test_corrupt_pdf_gives_a_422_not_a_500(self, client):
        response = client.post(
            "/upload",
            files={"file": ("broken.pdf", b"%PDF-1.4 truncated garbage", "application/pdf")},
        )
        assert response.status_code == 422

    def test_textless_file_is_rejected_with_an_explanation(self, client):
        response = client.post("/upload", files={"file": ("blank.txt", b"    \n  ", "text/plain")})
        assert response.status_code == 422
        assert "OCR" in response.json()["detail"]

    def test_oversized_upload_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
        response = client.post("/upload", files={"file": ("big.txt", b"x" * 100, "text/plain")})
        assert response.status_code == 413
