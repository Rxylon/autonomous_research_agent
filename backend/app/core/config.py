from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
from os import getenv
from pathlib import Path

from dotenv import load_dotenv


_BACKEND_ROOT = Path(__file__).resolve().parents[2]

# ENV_FILE overrides which dotenv file is read. Pointing it at a path that does not
# exist makes loading a no-op, which is how the test suite keeps a developer's real
# backend/.env — and the live API keys in it — out of test runs.
_ENV_FILE = Path(getenv("ENV_FILE") or (_BACKEND_ROOT / ".env"))
load_dotenv(dotenv_path=_ENV_FILE, override=False)


def _cors_origins() -> list[str]:
    raw_value = getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def _resolve_path(value: str) -> str:
    """Resolve a configured path relative to the backend package root.

    Without this, ``data/chroma`` lands wherever uvicorn happened to be started
    from, so history and the vector store silently split across directories.
    """
    path = Path(value)
    return str(path if path.is_absolute() else _BACKEND_ROOT / path)


@dataclass(slots=True)
class Settings:
    app_name: str = "Autonomous Multi-Agent Research Assistant"
    environment: str = getenv("ENVIRONMENT", "development")
    cors_origins: list[str] = field(default_factory=_cors_origins)
    openai_api_key: str | None = getenv("OPENAI_API_KEY") or None
    gemini_api_key: str | None = getenv("GEMINI_API_KEY") or None
    default_llm_provider: str = getenv("DEFAULT_LLM_PROVIDER", "mock")
    default_llm_model: str = getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")
    # Which embedding tier to use: `auto` (local model, falling back to hashing),
    # `local` (Sentence-Transformers only), `hash` (no model at all), or `openai`.
    # `hash` exists for memory-constrained hosts: loading torch plus a transformer
    # needs several hundred MB, which does not fit in a 512 MB free-tier container.
    embedding_backend_preference: str = getenv("EMBEDDING_BACKEND", "auto").strip().lower()
    # Sentence-Transformers id used for local embeddings. Note the `-en-`: the
    # id without it does not exist on Hugging Face.
    embedding_model: str = getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # OpenAI embedding model. Kept separate from `embedding_model` because the two
    # id namespaces are not interchangeable — passing a Hugging Face id to OpenAI
    # fails every call.
    openai_embedding_model: str = getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_persist_directory: str = field(
        default_factory=lambda: _resolve_path(getenv("CHROMA_PERSIST_DIRECTORY", str(Path("data") / "chroma")))
    )
    chroma_collection_name: str = getenv("CHROMA_COLLECTION_NAME", "research_documents")
    data_directory: str = field(default_factory=lambda: _resolve_path(getenv("DATA_DIRECTORY", "data")))
    reports_directory: str = field(default_factory=lambda: _resolve_path(getenv("REPORTS_DIRECTORY", "reports")))
    # Cap on documents pulled per search query, to keep free-tier runs inside
    # request timeouts.
    max_documents_per_query: int = int(getenv("MAX_DOCUMENTS_PER_QUERY", "5"))
    http_timeout_seconds: float = float(getenv("HTTP_TIMEOUT_SECONDS", "8"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
