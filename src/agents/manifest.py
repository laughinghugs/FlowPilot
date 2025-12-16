"""Plan manifest persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from .llm import PlanStep


@dataclass(frozen=True)
class PlanManifestEntry:
    plan_id: str
    created_at: str
    user_message: str
    system_prompt: str | None
    steps: Sequence[PlanStep] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        user_message: str,
        steps: Sequence[PlanStep],
        system_prompt: str | None,
    ) -> "PlanManifestEntry":
        return cls(
            plan_id=str(uuid4()),
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            user_message=user_message,
            system_prompt=system_prompt,
            steps=list(steps),
        )

    def to_serializable(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "user_message": self.user_message,
            "system_prompt": self.system_prompt,
            "steps": [asdict(step) for step in self.steps],
        }


class PlanManifestWriter:
    """Append-only JSONL writer for agent plans."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def write(self, entry: PlanManifestEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            json_line = json.dumps(entry.to_serializable(), ensure_ascii=False)
            handle.write(json_line + "\n")
