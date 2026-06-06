from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
from os import getenv
from pathlib import Path

from dotenv import load_dotenv


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_BACKEND_ROOT / ".env", override=False)


def _cors_origins() -> list[str]:
    raw_value = getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str = "Autonomous Multi-Agent Research Assistant"
    environment: str = "development"
    cors_origins: list[str] = field(default_factory=_cors_origins)
    openai_api_key: str | None = getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = getenv("GEMINI_API_KEY")
    default_llm_provider: str = getenv("DEFAULT_LLM_PROVIDER", "mock")
    embedding_model: str = getenv("EMBEDDING_MODEL", "BAAI/bge-small-v1.5")
    chroma_persist_directory: str = getenv("CHROMA_PERSIST_DIRECTORY", str(Path("data") / "chroma"))
    chroma_collection_name: str = getenv("CHROMA_COLLECTION_NAME", "research_documents")
    data_directory: str = getenv("DATA_DIRECTORY", str(Path("data")))
    reports_directory: str = getenv("REPORTS_DIRECTORY", str(Path("reports")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
