# Autonomous Multi-Agent Research Assistant

> Turns one research question into a five-stage pipeline — plan, retrieve, summarize, verify, report — with live progress over WebSocket and every claim traced back to a real paper.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)
![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen)

**Live demo:** [research-ai-assistant-e5306.web.app](https://research-ai-assistant-e5306.web.app) · **API:** [ai-research-backend-y7av.onrender.com/docs](https://ai-research-backend-y7av.onrender.com/docs)

> The demo runs on free tiers and is **noticeably degraded** compared to running locally. Read [Live demo vs. local](#live-demo-vs-local) before judging it — the first request after idle takes ~50 s, and depending on the deployed key state, summaries may come from a non-LLM fallback.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Live demo vs. local](#live-demo-vs-local)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Design notes and known limitations](#design-notes-and-known-limitations)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## What It Does

Submit a research question. The system runs five stages and streams each one to the browser as it happens:

| Stage | What it actually does | Uses an LLM? |
|---|---|---|
| **Planner** | Strips conversational filler from the query and expands it into 3 search variants. The step list is a constant. | No — rule-based |
| **Retrieval** | Searches arXiv and Semantic Scholar in parallel (Crossref as fallback), indexes results into ChromaDB, then queries that index to recall passages from uploaded documents and earlier runs | No |
| **Summarizer** | Synthesises the sources into Markdown bullets covering approaches, findings, and open problems | Yes |
| **Critic** | Breaks the *published summary* into claims and checks each against the retrieved sources, returning a 0–1 score | Yes |
| **Reporter** | Renders Markdown, JSON, and PDF artifacts | No |

Every response reports **how** the critic scored it — `llm` (a model checked the claims), `heuristic` (keyword overlap only), or `empty` (nothing was checked). A 1.00 from the heuristic path is not evidence of anything, and the UI labels it as such rather than showing a bare number.

### Example

Query: *"Find recent advances in multimodal deepfake detection and summarize major approaches."*

Returns a research plan, ~6–12 papers with abstracts and URLs, a synthesis, per-claim verification with cited source indices, and Markdown/JSON/PDF downloads. A real local run against arXiv retrieved:

```
[planning    ] complete  Planner generated a research plan.
[retrieval   ] complete  Retrieved 6 source documents.
[summarizing ] complete  Summary generated.
[critic      ] complete  Critic score: 0.64
[report      ] complete  Report generated.
```

---

## Live demo vs. local

Both tiers are free, and the trade-offs are structural rather than bugs. **Everything below works fully when you run it locally.**

| Capability | Local | Live demo |
|---|---|---|
| Full 5-stage pipeline | ✅ | ✅ |
| WebSocket live progress | ✅ | ✅ (Render supports WebSockets) |
| First-request latency | ~1 s | **~50 s** — free instances sleep after 15 min idle and cold-start the whole Docker image |
| LLM summarization + claim checking | ✅ with a key | Depends on the key set in the Render dashboard. If absent, invalid, or rate-limited, summaries fall back to a **mechanical listing** and scores become keyword heuristics. `GET /health` reports this in `llm_configured` and `last_llm_error`, and the UI shows a banner. |
| Semantic vector search | ✅ Sentence-Transformers (`bge-small-en-v1.5`), via `requirements-local.txt` | ❌ **Lexical only.** 512 MB RAM cannot hold torch plus a transformer, so the image omits both and pins `EMBEDDING_BACKEND=hash` — a deterministic hashed bag-of-words. Retrieval still works but matches on shared tokens, not meaning. |
| Run history | ✅ persists in `data/history.jsonl` | ⚠️ **Resets on restart.** The filesystem is ephemeral, so History, Reports, and Analytics empty out whenever the instance sleeps. |
| Uploaded documents (`POST /upload`) | ✅ persist and influence later queries | ⚠️ Work within a session; the Chroma index is lost on restart |
| Markdown / JSON report download | ✅ | ✅ while the run is still in history (served from history, not disk) |
| **PDF report download** | ✅ | ❌ **Returns 410 after a restart** — the PDF exists only on the ephemeral disk and cannot be reconstructed from history |
| Concurrency | Fine | One free instance, single worker. Two simultaneous runs queue behind each other. |

### On Vercel specifically

The repo deploys the backend to **Render** (`render.yaml`), and the live API above is a Render service. That is deliberate: **Vercel's serverless functions do not support WebSockets**, so `/ws/research` — the live-progress feature — cannot work there. A Vercel deployment would also hit the function execution time limit, since one research run takes 10–60 s. If you want the backend on Vercel, the streaming endpoint has to be dropped and the frontend has to fall back to `POST /query`; the frontend already does this automatically when the WebSocket fails.

---

## Quick Start

### Prerequisites

- **Python 3.10+**, **Node.js 18+**
- An **OpenAI** or **Gemini** API key (optional — see below)

Without a key the app still runs end to end. Retrieval, verification plumbing, reporting, downloads, and streaming all work; only the summarization and claim-checking *content* degrades to a non-LLM fallback, which labels itself as such.

### Backend

```bash
cd backend

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# Core install — full pipeline, lexical (hashed) embeddings. ~200 MB.
pip install -r requirements.txt

# Optional: add semantic embeddings (Sentence-Transformers + torch, ~2 GB).
# pip install -r requirements-local.txt

cp .env.example .env              # then edit .env
uvicorn app.main:app --reload --port 8000
```

Both installs run the complete five-stage pipeline. The only difference is retrieval quality: `requirements.txt` alone matches on shared tokens, `requirements-local.txt` matches on meaning. `GET /health` reports which tier is active under `embedding_backend`, so you never have to guess.

Verify what is actually live — this is the fastest way to catch a misconfigured key:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "llm_provider": "gemini",
  "llm_configured": true,
  "llm_model": "gemini-2.0-flash",
  "embedding_backend": "sentence-transformers",
  "vector_store_documents": 128,
  "last_llm_error": null
}
```

`llm_configured: false` means every summary will come from the local fallback. `last_llm_error` being non-null while `llm_configured` is true means a key is present but calls are failing — the failure mode that otherwise looks identical to success.

Swagger docs: `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local        # defaults to http://localhost:8000
npm run dev
```

Open `http://localhost:3000`, enter a question, and press **Run research** (or `⌘↵`).

> `NEXT_PUBLIC_BACKEND_URL` is inlined at **build** time. Changing it needs a rebuild, not just a restart.

---

## Architecture

```
                                    ┌─────────────────────────┐
User query ──► POST /query          │  ChromaDB (persistent)  │
           └─► WS /ws/research      │  uploads + past runs    │
                     │              └────┬───────────────▲────┘
                     ▼                   │ recall        │ index
        ┌────────────────────────────────┼───────────────┼────┐
        │  LangGraph StateGraph          │               │    │
        │                                │               │    │
        │  planner ─► retrieval ─────────┴───────────────┘    │
        │               │                                     │
        │               ▼                                     │
        │            summary ─► critic ─► report ─► END        │
        └─────────────────────────────────────────────────────┘
                     │                          │
                     ▼                          ▼
        progress events (WebSocket)     history.jsonl + reports/
```

**The graph is the execution path.** `run_with_progress` drives `compiled.astream(...)`, so the edges above determine order. (An earlier version compiled the graph and then called each node function by hand, which meant the graph was decoration — `GET /config` now reports `engine` so you can confirm which path is live.)

**Nothing blocking runs on the event loop.** Each node is `async`, but the work inside — HTTP search, LLM calls, embedding, PDF rendering — is synchronous and slow, so each is dispatched with `asyncio.to_thread`. Retrieval additionally fans its network calls out across a thread pool, so searching two backends across three query variants costs roughly one round trip rather than six.

**The vector store is read, not just written.** Retrieval indexes fresh search results *and then queries the index*. That read is what lets a PDF you uploaded, or a paper found by an earlier run, surface in a later answer.

**The critic grades the published summary.** It receives the exact text the user sees, in a separate call from the summarizer. Grading an independently redrafted summary would say nothing about what was published.

---

## Configuration

`backend/.env` — see `backend/.env.example` for the annotated version.

| Variable | Default | Notes |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `mock` | `gemini`, `openai`, or `mock` |
| `DEFAULT_LLM_MODEL` | `gemini-2.0-flash` | `gemini-2.0-flash`, `gemini-2.5-flash`, `gpt-4o-mini`, … |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | Only the one for your provider |
| `EMBEDDING_BACKEND` | `auto` | `auto`, `local`, `hash`, `openai`. Use `hash` under ~2 GB RAM. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Note the `-en-`; the id without it does not exist |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated |
| `CHROMA_PERSIST_DIRECTORY` | `data/chroma` | Relative paths resolve against `backend/`, not the shell's cwd |
| `DATA_DIRECTORY` / `REPORTS_DIRECTORY` | `data` / `reports` | |
| `MAX_DOCUMENTS_PER_QUERY` | `5` | Per search query, per backend |
| `HTTP_TIMEOUT_SECONDS` | `8` | Paper-search timeout |
| `ENV_FILE` | `backend/.env` | Point at a nonexistent path to skip dotenv (the test suite does this) |

`frontend/.env.local`:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

Switching `EMBEDDING_BACKEND` starts a **fresh Chroma collection** — the collection name embeds the active backend, because vectors from different models are not comparable and OpenAI's are a different width. Nothing is corrupted; previously indexed documents simply are not visible to the new backend.

---

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Run the pipeline synchronously |
| `WS` | `/ws/research` | Run it with live progress events |
| `POST` | `/upload` | Index a PDF or text file into the vector store |
| `GET` | `/history?limit=25` | List prior runs, newest first |
| `GET` | `/history/{run_id}` | One run, with full report |
| `GET` | `/reports/{run_id}.md` | Download Markdown (from history) |
| `GET` | `/reports/{run_id}.json` | Download JSON (from history) |
| `GET` | `/reports/{run_id}.pdf` | Download PDF (from disk; `410` if lost) |
| `GET` | `/health` | Provider, key status, embedding tier, last provider error |
| `GET` | `/config` | Non-secret runtime config, incl. `engine` |

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain attention mechanisms in transformers", "top_k": 5}'
```

```json
{
  "run_id": "49ca1082-da24-4890-bb2d-782c7cfaa455",
  "status": "complete",
  "plan": { "objective": "...", "steps": ["..."], "search_queries": ["..."] },
  "summary": "- ...",
  "critic_score": 0.64,
  "critic_method": "llm",
  "claim_checks": [
    {
      "claim": "Scaled dot-product attention is the core operation",
      "supported": true,
      "evidence": ["[1] Attention Is All You Need — We propose a new simple network..."],
      "rationale": "Stated directly in source 1."
    }
  ],
  "report_markdown": "# Research Report\n...",
  "sources": [{ "title": "...", "url": "...", "year": 2024, "abstract": "...", "source": "arxiv" }],
  "error": null
}
```

A failed run returns `status: "failed"` with `error` set, and is still written to history — a run never disappears silently.

### `WS /ws/research`

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/research");
ws.onopen = () => ws.send(JSON.stringify({ query: "Neural architecture search advances" }));
ws.onmessage = (event) => {
  const e = JSON.parse(event.data);
  if (e.type === "progress") console.log(`[${e.stage}] ${e.status} — ${e.message}`);
  if (e.type === "result") console.log("score:", e.result.critic_score, e.result.critic_method);
  if (e.type === "error") console.error(e.message);
};
```

Stages arrive as `queued → planning → retrieval → summarizing → critic → report`, each `running` then `complete`, followed by one `result` (or `error`).

### `POST /upload`

```bash
curl -F "file=@paper.pdf" http://localhost:8000/upload
```

PDFs are parsed with PyMuPDF. Scanned PDFs with no text layer return `422` — this service does not do OCR. Limit is 20 MB. Uploaded content is indexed and **recalled into later queries** whose objective is semantically close.

---

## Design notes and known limitations

Deliberate trade-offs, stated plainly rather than implied away.

1. **The planner does not use an LLM.** It strips filler and produces three search variants; the step list is a constant. This costs nothing, adds no latency, and never fails, which matters when a whole run has to fit inside one free-tier request timeout. The cost is that planning does not adapt to the question beyond keyword extraction.

2. **The critic and summarizer are two separate LLM calls.** That doubles token cost per run. It is the point: one call writes, a second grades what was written. Collapsing them would mean the model scoring its own draft in the same breath as producing it.

3. **The critic's score is only as good as the model's judgement.** It is the fraction of extracted claims the model marked supported. A model that is lenient produces a high score. `critic_method` tells you whether a model was involved at all; treat `heuristic` scores as a liveness signal, not verification.

4. **The fallback summary is a listing, not a synthesis.** With no working provider, output enumerates what was retrieved and says so in its own text. It is honest, not useful.

5. **Retrieval reads abstracts, not full papers.** Both search APIs return abstracts, and that is what gets indexed and summarized. Full-text PDF extraction exists (`extract_pdf_text`) and is wired to `POST /upload`, but the pipeline does not fetch and parse PDFs for papers it finds itself.

6. **`history.jsonl` is append-only and read in full on every request.** Fine for hundreds of runs, greppable while debugging, wrong at scale. Reads are O(runs).

7. **Hashed embeddings are a real fallback, not a silent one.** `EMBEDDING_BACKEND=hash` matches on shared tokens only. `GET /health` reports which tier is active.

8. **CORS does not cover WebSockets.** Browsers send no preflight for WS upgrades and Starlette's `CORSMiddleware` does not filter them, so `/ws/research` is reachable from any origin regardless of `CORS_ORIGINS`.

9. **There is no auth, and no rate limiting.** Anyone who can reach the backend can spend your LLM quota. Do not expose a keyed deployment publicly without putting something in front of it.

10. **A rotated key is still required.** A live Gemini key was previously committed to `backend/.env.example` and is in this repository's public git history. It has been removed from the working tree, but **git history is not rewritten**, so that key must be treated as compromised and rotated. Anything in `.env` never entered history and is unaffected.

---

## Testing

145 pytest tests, none of which touch the network — provider and search calls are stubbed, so the suite is deterministic and free to run.

```bash
cd backend
python -m pytest tests/ -q
```

```
145 passed in 59s
```

Coverage is organised around the regressions that motivated it: that the compiled graph is actually driven, that nodes leave the event loop, that a `supported: false` verdict is never overturned, that the vector store round-trips, that a PDF upload extracts real text, that naive legacy timestamps do not break history sorting.

**Live end-to-end check** — hits arXiv, Semantic Scholar, and your configured LLM. Takes 10–60 s and spends tokens:

```bash
cd backend
python scripts/smoke_e2e.py
python scripts/smoke_e2e.py --query "Explain attention mechanisms in transformers"
```

Exits non-zero if the run does not complete, so it works in CI.

**Frontend:**

```bash
cd frontend
npm run lint        # clean
npx tsc --noEmit    # clean
npm run build       # static export to out/
```

---

## Deployment

### Frontend → Firebase Hosting

`next.config.ts` sets `output: "export"`, so the build is a **static bundle — there is no SSR or server-side rendering at runtime.** All six routes prerender to HTML.

```bash
cd frontend
npm run build          # -> out/
firebase deploy --only hosting
```

### Backend → Render (Docker)

`render.yaml` defines the service; it deploys on push once the blueprint is connected. Set `GEMINI_API_KEY` in the Render dashboard — never in the repo.

```bash
git push origin master
```

The image installs `requirements.txt` only, so it contains **no torch and no sentence-transformers**, and `EMBEDDING_BACKEND=hash` is set in both the Dockerfile and `render.yaml`. That is what makes the image small enough to build and start on a free instance.

If you need semantic embeddings in a container, switch the Dockerfile to `requirements-local.txt` **and** read the index-URL note in that file. Installing CPU torch with only `--index-url https://download.pytorch.org/whl/cpu` breaks the build: that flag *replaces* PyPI rather than adding to it, so any package pip has to build from source cannot resolve its build backend, and the layer fails with `No matching distribution found for flit_core`. Pass `--extra-index-url https://pypi.org/simple` alongside it.

---

## Project Structure

```
autonomous_multi_research_agent/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── planner.py      # rule-based query expansion
│   │   │   ├── retrieval.py    # search + index + vector recall; PDF extraction
│   │   │   ├── summarizer.py   # LLM synthesis
│   │   │   ├── critic.py       # claim checking against sources
│   │   │   └── report.py       # Markdown / JSON / PDF rendering
│   │   ├── api/routes.py       # endpoints, WebSocket, downloads
│   │   ├── core/config.py      # settings & env
│   │   ├── graph/workflow.py   # LangGraph orchestration
│   │   ├── models/schemas.py   # Pydantic models
│   │   ├── services/
│   │   │   ├── llm_provider.py # OpenAI / Gemini + local fallback
│   │   │   ├── vector_store.py # ChromaDB read + write
│   │   │   ├── embeddings.py   # openai | sentence-transformers | hash
│   │   │   ├── paper_search.py # arXiv / Semantic Scholar / Crossref
│   │   │   └── history_store.py
│   │   └── main.py
│   ├── scripts/smoke_e2e.py    # live end-to-end check
│   ├── tests/                  # 145 pytest tests
│   ├── data/                   # history.jsonl + chroma/ (gitignored)
│   ├── reports/                # generated artifacts (gitignored)
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/                # /, /history, /reports, /reports/detail, /dashboard
│   │   ├── components/
│   │   │   ├── research-console.tsx   # prompt, stream, results, downloads
│   │   │   └── ui/
│   │   ├── services/api.ts     # typed client; propagates errors
│   │   ├── data/stages.ts      # stage placeholder copy
│   │   └── types/research.ts
│   ├── firebase.json
│   └── next.config.ts
│
├── render.yaml
├── LICENSE
└── README.md
```

---

## Contributing

1. Fork and branch: `git checkout -b feature/my-feature`
2. Make changes, add tests for behaviour you rely on
3. `cd backend && python -m pytest tests/ -q`
4. `cd frontend && npm run lint && npm run build`
5. Open a PR

**Code style:** PEP 8 for Python (`black`-compatible), the project's ESLint config for TypeScript, conventional commit prefixes (`feat:`, `fix:`, `docs:`).

Never commit a real API key. `.env.example` is tracked; `.env` is not.

---

## Roadmap

- [ ] Full-text PDF fetching for discovered papers, not just abstracts
- [ ] LLM-driven planning as an opt-in mode
- [ ] Local LLM support (Ollama, llama.cpp)
- [ ] Auth and per-key rate limiting
- [ ] Incremental report updates without re-running the pipeline
- [ ] BibTeX / RIS export

---

## License

MIT — see [LICENSE](LICENSE).
