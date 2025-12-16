"""Core primitives for composing agent-style responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable


@dataclass(frozen=True)
class AgentMessage:
    """Simple, typed container for agent messages."""

    role: str
    content: str

    def pretty(self) -> str:
        """Return the message as a CLI-friendly string."""
        return f"[{self.role}] {self.content.strip()}"


def build_agent_response(messages: Iterable[AgentMessage]) -> str:
    """Create a deterministic agent response from a history of messages."""

    history = [message.pretty() for message in messages]
    timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")

    if not history:
        return f"[{timestamp}] No context provided."

    summary = " | ".join(history)
    return f"[{timestamp}] Context: {summary}"
