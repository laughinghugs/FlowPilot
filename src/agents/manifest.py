"""Plan manifest persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from .llm import CustomToolDefinition, PlanStep


@dataclass(frozen=True)
class PlanManifestEntry:
    plan_id: str
    created_at: str
    user_message: str
    system_prompt: str | None
    steps: Sequence[PlanStep] = field(default_factory=list)
    custom_tools: Sequence[CustomToolDefinition] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        user_message: str,
        steps: Sequence[PlanStep],
        system_prompt: str | None,
        custom_tools: Sequence[CustomToolDefinition] | None = None,
    ) -> "PlanManifestEntry":
        return cls(
            plan_id=str(uuid4()),
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            user_message=user_message,
            system_prompt=system_prompt,
            steps=list(steps),
            custom_tools=list(custom_tools or []),
        )

    def to_serializable(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "user_message": self.user_message,
            "system_prompt": self.system_prompt,
            "steps": [asdict(step) for step in self.steps],
            "custom_tools": [asdict(tool) for tool in self.custom_tools],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "PlanManifestEntry":
        steps = [PlanStep(**step) for step in data.get("steps", [])]
        return cls(
            plan_id=data["plan_id"],
            created_at=data["created_at"],
            user_message=data["user_message"],
            system_prompt=data.get("system_prompt"),
            steps=steps,
            custom_tools=[CustomToolDefinition(**tool) for tool in data.get("custom_tools", [])],
        )


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


class PlanManifestReader:
    """Reader utility for looking up manifest entries by id."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def get(self, plan_id: str) -> PlanManifestEntry:
        if not self._path.exists():
            raise FileNotFoundError(f"Manifest file '{self._path}' not found")

        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("plan_id") == plan_id:
                    return PlanManifestEntry.from_serializable(data)

        raise ValueError(f"Plan id '{plan_id}' not found in manifest '{self._path}'")
