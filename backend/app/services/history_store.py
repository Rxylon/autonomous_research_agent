"""Append-only JSONL store for research runs.

Chosen for legibility over a database: one run per line, greppable, and trivially
inspectable while debugging. The trade-offs are real and worth stating — reads scan
the whole file, and on a host with an ephemeral filesystem the file is lost on
restart. Both are documented in the README rather than papered over.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.models.schemas import ResearchRun


def _as_aware(value: datetime) -> datetime:
    """Force a timestamp to UTC-aware.

    Runs written before the move off ``datetime.utcnow()`` have naive timestamps.
    Sorting a mix of naive and aware datetimes raises TypeError, so every value read
    back is normalised — naive ones are interpreted as the UTC they were meant to be.
    """
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class HistoryStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else Path(settings.data_directory) / "history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, run: ResearchRun) -> None:
        payload = run.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_all(self) -> list[ResearchRun]:
        if not self.path.exists():
            return []

        runs: list[ResearchRun] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    run = ResearchRun.model_validate_json(line)
                except Exception:
                    # A single malformed or older-schema line must not make the whole
                    # history endpoint fail.
                    continue
                run.created_at = _as_aware(run.created_at)
                run.updated_at = _as_aware(run.updated_at)
                runs.append(run)
        return runs

    def list(self, limit: int = 20) -> list[ResearchRun]:
        return sorted(self._read_all(), key=lambda run: run.created_at, reverse=True)[:limit]

    def get(self, run_id: UUID) -> ResearchRun | None:
        for run in reversed(self._read_all()):
            if run.id == run_id:
                return run
        return None
