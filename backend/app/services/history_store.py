from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.models.schemas import ResearchRun


class HistoryStore:
    def __init__(self) -> None:
        self.path = Path(settings.data_directory) / "history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, run: ResearchRun) -> None:
        payload = run.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list(self, limit: int = 20) -> list[ResearchRun]:
        if not self.path.exists():
            return []
        runs: list[ResearchRun] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                runs.append(ResearchRun.model_validate_json(line))
        return sorted(runs, key=lambda item: item.created_at, reverse=True)[:limit]

    def get(self, run_id: UUID) -> ResearchRun | None:
        for run in self.list(limit=500):
            if run.id == run_id:
                return run
        return None
