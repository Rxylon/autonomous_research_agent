# Autonomous Multi-Agent Research Assistant

> A full-stack research system that turns a single research prompt into a multi-stage workflow with planning, retrieval, summarization, verification, and reporting—all with live progress streaming and source-backed outputs.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)

## Table of Contents

- [What It Does](#what-it-does)
- [Why This Project](#why-this-project)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Support](#support)

## What It Does

The **Autonomous Multi-Agent Research Assistant** automates academic research workflows. Submit a research question, and the system orchestrates a five-stage pipeline:

1. **Planner** — Breaks your query into executable research steps and search strategies
2. **Retrieval** — Searches arXiv, Semantic Scholar, and other academic sources  
3. **Summarizer** — Synthesizes findings into major themes, approaches, and open problems
4. **Critic** — Validates claims against source evidence and scores confidence (0–1)
5. **Reporter** — Exports structured reports in Markdown, JSON, and PDF

All stages stream live progress via WebSocket, and all citations are traced back to source documents.

### Example Output

Submit: *"Find recent advances in multimodal deepfake detection and summarize major approaches."*

Get back:
- ✅ Structured research plan with search queries
- ✅ 15+ peer-reviewed sources with full abstracts
- ✅ Synthesis of major methods (next-frame prediction, multimodal fusion, transformer architectures)
- ✅ Critic score: 1.0 (all major claims supported by sources)
- ✅ Downloadable Markdown, JSON, and PDF reports

---

## Why This Project

### The Problem
- **Manual research is slow** — Hours to find, read, and synthesize papers
- **Hallucinations are risky** — LLMs can fabricate citations and claims
- **Verification is tedious** — No automatic fact-checking against source evidence
- **Reports aren't reproducible** — Hard to trace claims back to specific papers

### Our Solution
- **End-to-end automation** — From query to verified report in minutes
- **Source-backed claims** — Every assertion is validated against real papers with URLs
- **Transparency** — WebSocket streaming shows each stage in real time
- **Production-ready exports** — Markdown, JSON, and PDF with full citation metadata

---

## Quick Start

### Prerequisites
- **Python 3.10+** (backend)
- **Node.js 18+** (frontend)
- **pip** and **npm**
- _(Optional)_ **OpenAI API key** or **Gemini API key** for LLM support

### 1. Clone and Setup

```bash
git clone <repository-url>
cd autonomous_multi_research_agent
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Or on macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy and edit)
cp .env.example .env

# Start the server
uvicorn app.main:app --reload --port 8000
```

The backend listens on `http://localhost:8000` with Swagger docs at `/docs`.

### 3. Frontend Setup

In a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open `http://localhost:3000` in your browser.

### 4. Run a Research Query

1. Navigate to the **Chat** tab
2. Enter a research question (e.g., *"Explain recent advances in neural text generation"*)
3. Click **Run research**
4. Watch live progress stream through planning → retrieval → summarization → critic → report
5. View the critic score and download reports from the **Reports** tab

---

## Architecture

### Data Flow

```
User Query
    ↓
[Planner] — Decomposes into search steps
    ↓
[Retrieval] — Fetches papers from arXiv/Semantic Scholar
    ↓
[Vector Store] — Embeds documents in ChromaDB
    ↓
[Summarizer] — Synthesizes findings into themes
    ↓
[Critic] — Validates claims with confidence score
    ↓
[Reporter] — Generates Markdown/JSON/PDF
    ↓
User Downloads Report + History
```

### Multi-Agent Stages

| Stage | Role | Input | Output |
|-------|------|-------|--------|
| **Planner** | Decompose query | Research objective | Search queries, research steps |
| **Retrieval** | Find papers | Research plan | Sources with abstracts, full text |
| **Summarizer** | Synthesize findings | Plan + documents | Summary text, major themes |
| **Critic** | Verify claims | Summary + sources | Confidence score (0–1), claim checks |
| **Reporter** | Export results | All stages | Markdown, JSON, PDF artifacts |

---

## Tech Stack

### Frontend
- **Next.js 16** — React framework with SSR and built-in optimization
- **TypeScript** — Type-safe JavaScript
- **Tailwind CSS 4** — Utility-first styling
- **TanStack React Query** — Server state management
- **Framer Motion** — Smooth animations
- **Lucide React** — Icon library

### Backend
- **FastAPI** — Modern, fast Python web framework
- **LangGraph** — Multi-agent orchestration and state management
- **LangChain** — LLM integrations (OpenAI, Gemini)
- **ChromaDB** — Local vector database
- **Sentence Transformers** — Embedding models (BAAI/bge-small-v1.5)
- **arxiv + Semantic Scholar APIs** — Academic paper discovery
- **ReportLab** — PDF generation

### Data & Storage
- **JSONL** — Research run history (local `data/history.jsonl`)
- **ChromaDB** — Vector search index (`data/chroma/`)
- **SQLite** — ChromaDB metadata backend

---

## Usage Examples

### Example 1: Research Query via UI

1. **Open** `http://localhost:3000`
2. **Enter query**: *"Compare graph neural networks vs transformer architectures for link prediction"*
3. **Watch live updates** as each stage completes
4. **Download report** as PDF or view in Markdown

### Example 2: API Query via cURL

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain attention mechanisms in transformers"}'
```

Response:
```json
{
  "run_id": "uuid-here",
  "status": "complete",
  "critic_score": 0.95,
  "summary": "...",
  "sources": [...],
  "report_markdown": "# Research Report\n..."
}
```

### Example 3: WebSocket Streaming (Live Progress)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/research');

ws.onopen = () => {
  ws.send(JSON.stringify({ 
    query: "Find recent advances in neural architecture search" 
  }));
};

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`[${progress.stage}] ${progress.message}`);
  if (progress.type === 'result') {
    console.log(`Final critic score: ${progress.result.critic_score}`);
  }
};
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Provider (openai, gemini, or mock)
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.0-flash

# API Keys (leave empty to use mock/local fallback)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Embedding Model (from Hugging Face)
EMBEDDING_MODEL=BAAI/bge-small-v1.5

# Storage Paths
CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=research_documents
DATA_DIRECTORY=data
REPORTS_DIRECTORY=reports

# CORS (frontend origin)
CORS_ORIGINS=http://localhost:3000

# Frontend Backend URL (if not localhost:8000)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Switching LLM Providers

**Option 1: OpenAI**
```bash
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key
```

**Option 2: Google Gemini**
```bash
DEFAULT_LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
DEFAULT_LLM_MODEL=gemini-2.0-flash
```

**Option 3: Mock (local fallback, no keys needed)**
```bash
DEFAULT_LLM_PROVIDER=mock
```

---

## API Reference

### Endpoints

#### `POST /query`
Submit a synchronous research query.

**Request:**
```json
{ "query": "Explain transformer attention mechanisms" }
```

**Response:**
```json
{
  "run_id": "string (UUID)",
  "status": "complete|failed",
  "plan": { "objective": "...", "steps": [...], "search_queries": [...] },
  "summary": "Markdown summary",
  "critic_score": 0.0,
  "report_markdown": "# Research Report\n...",
  "sources": [{ "title": "...", "url": "...", "year": 2024, "abstract": "..." }],
  "error": null
}
```

#### `WebSocket /ws/research`
Stream live progress events.

**Send:**
```json
{ "query": "Your research question" }
```

**Receive (progress events):**
```json
{
  "type": "progress",
  "stage": "planning|retrieval|summarizing|critic|report",
  "status": "running|complete",
  "message": "Human-readable update",
  "details": { ... }
}
```

**Receive (final result):**
```json
{
  "type": "result",
  "stage": "complete",
  "message": "Research run completed",
  "result": { ... }  // Same as POST /query response
}
```

#### `GET /history`
List all prior research runs (limit 25).

**Response:**
```json
[
  {
    "id": "uuid",
    "query": "...",
    "status": "complete",
    "created_at": "2026-05-30T...",
    "summary": "...",
    "critic_score": 0.95
  }
]
```

#### `GET /history/{run_id}`
Retrieve a specific run with full report.

#### `POST /upload`
Upload a custom document to the vector store.

**Form data:**
- `file` — PDF or text file

**Response:**
```json
{
  "document_id": "string",
  "filename": "string",
  "chunks_indexed": 42,
  "metadata": { "content_length": 50000 }
}
```

#### `GET /health`
Health check.

**Response:**
```json
{ "status": "ok" }
```

---

## Project Structure

```
autonomous_multi_research_agent/
├── backend/
│   ├── app/
│   │   ├── agents/           # Multi-agent implementations
│   │   │   ├── planner.py    # Query decomposition
│   │   │   ├── retrieval.py  # Paper search & fetching
│   │   │   ├── summarizer.py # Synthesis
│   │   │   ├── critic.py     # Claim validation
│   │   │   └── report.py     # Report generation
│   │   ├── api/
│   │   │   └── routes.py     # FastAPI endpoints
│   │   ├── core/
│   │   │   └── config.py     # Settings & environment
│   │   ├── graph/
│   │   │   └── workflow.py   # LangGraph orchestration
│   │   ├── models/
│   │   │   └── schemas.py    # Pydantic models
│   │   ├── services/
│   │   │   ├── llm_provider.py      # LLM abstraction
│   │   │   ├── vector_store.py      # ChromaDB wrapper
│   │   │   ├── history_store.py     # Run persistence
│   │   │   ├── paper_search.py      # arXiv/Semantic Scholar
│   │   │   └── embeddings.py        # Embedding service
│   │   └── main.py           # FastAPI app
│   ├── data/
│   │   ├── history.jsonl     # Research run history
│   │   └── chroma/           # Vector store
│   ├── reports/              # Generated PDF/Markdown reports
│   ├── requirements.txt      # Python dependencies
│   └── .env                  # Configuration (create from .env.example)
│
├── frontend/
│   ├── src/
│   │   ├── app/              # Next.js pages
│   │   │   ├── page.tsx      # Home/chat interface
│   │   │   ├── layout.tsx    # Root layout
│   │   │   ├── reports/      # Report views
│   │   │   ├── history/      # History page
│   │   │   └── dashboard/    # Analytics
│   │   ├── components/       # Reusable components
│   │   │   ├── research-console.tsx   # Query input & streaming
│   │   │   ├── research-history.tsx   # Prior runs list
│   │   │   └── ui/           # Shadcn/ui components
│   │   ├── services/
│   │   │   └── api.ts        # Backend API client
│   │   ├── types/
│   │   │   └── research.ts   # TypeScript interfaces
│   │   └── data/
│   │       └── mock.ts       # Fallback mock data
│   ├── package.json          # Node dependencies
│   ├── next.config.ts        # Next.js config
│   └── tsconfig.json         # TypeScript config
│
└── README.md                 # This file
```

---

## Running Tests

### Backend Unit Tests

```bash
cd backend
python -m pytest tests/
```

### Manual Integration Test (WebSocket)

```bash
cd backend
python tests/ws_test_env.py
```

This runs a live research query via WebSocket and validates the output.

---

## Development Guide

### Adding a New Agent

1. Create `backend/app/agents/my_agent.py`:
   ```python
   from app.models.schemas import ResearchState
   
   class MyAgent:
       def invoke(self, state: ResearchState) -> dict:
           # Process state and return updates
           return {"new_field": result}
   ```

2. Import in `backend/app/graph/workflow.py`:
   ```python
   from app.agents.my_agent import MyAgent
   self.my_agent = MyAgent()
   ```

3. Add to workflow:
   ```python
   self._state_graph.add_node("my_stage", self._my_node)
   self._state_graph.add_edge("prior_stage", "my_stage")
   ```

### Debugging

Enable verbose logging:
```bash
cd backend
LOGLEVEL=DEBUG uvicorn app.main:app --reload
```

Check backend logs:
- `backend/logs/llm_provider_trace.log` — LLM provider resolution
- `backend/data/history.jsonl` — Full run history with payloads

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** and test locally
4. **Commit** with clear messages: `git commit -m "Add new agent for X"`
5. **Push** to your fork and open a **Pull Request**

### Code Style
- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/), use `black` for formatting
- **TypeScript**: Use ESLint config from `frontend/eslint.config.mjs`
- **Commits**: Use conventional commits (`feat:`, `fix:`, `docs:`, etc.)

### Before Submitting PR
- Run tests: `pytest backend/`
- Check linting: `npm run lint` (frontend)
- Update README if adding features

---

## Support & Documentation

### Getting Help

- **Issues**: [GitHub Issues](../../issues) — Report bugs or request features
- **Discussions**: Check existing issues before creating new ones
- **Docs**: See `backend/app/` for inline docstrings and schemas

### Common Issues

**Q: Backend returns critic_score: 0 and "Summary produced by local fallback"**

A: Environment variables not loaded. Ensure `.env` is in `backend/` and contains your API key:
```bash
cd backend
cat .env | grep GEMINI_API_KEY  # Should not be empty
```

**Q: WebSocket connection fails with CORS error**

A: Check `CORS_ORIGINS` in `.env` matches your frontend URL:
```bash
CORS_ORIGINS=http://localhost:3000  # For dev
```

**Q: Frontend shows "Request failed with 404"**

A: Backend not running. Start it first:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Roadmap

- [ ] Support for local LLMs (Ollama, LLaMA.cpp)
- [ ] Incremental report updates (don't re-run full pipeline)
- [ ] Custom agent templates for domain-specific research
- [ ] Multi-language support (non-English paper discovery)
- [ ] Collaborative research sessions (shared report editing)
- [ ] Export to bibliography formats (BibTeX, RIS)

---

**Built with ❤️ for researchers, developers, and curious minds.**

## Environment variables

Copy the example files and fill in API keys when you want live model access.

- `frontend/.env.example`
- `backend/.env.example`

## Validation

- Frontend production build passes
- Backend query smoke test passes
- Backend WebSocket progress stream passes
