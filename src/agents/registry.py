"""Tool registry and metadata primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ToolCapability:
    """Describe what a tool can do."""

    name: str
    category: str
    description: str


class ToolRegistry:
    """Registry that stores tool capabilities and allows user extensions."""

    def __init__(self, capabilities: Iterable[ToolCapability] | None = None) -> None:
        self._capabilities: list[ToolCapability] = list(capabilities or [])

    def register(self, *, name: str, category: str, description: str) -> ToolCapability:
        capability = ToolCapability(name=name, category=category, description=description)
        self._capabilities.append(capability)
        return capability

    def capabilities(self) -> Sequence[ToolCapability]:
        return tuple(self._capabilities)

    def categories(self) -> set[str]:
        return {capability.category for capability in self._capabilities}

    @classmethod
    def with_default_tools(cls) -> "ToolRegistry":
        """Load registry defaults from the packaged JSON resource."""

        return cls._from_resource("default_tools.json")

    @classmethod
    def from_json(cls, path: str) -> "ToolRegistry":
        """Create a registry from a JSON file on disk."""

        with open(path, "r", encoding="utf-8") as handle:
            specs = json.load(handle)
        return cls._from_specs(specs)

    @classmethod
    def _from_resource(cls, resource_name: str) -> "ToolRegistry":
        try:
            data = resources.files(__package__).joinpath(resource_name).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Cannot locate registry resource '{resource_name}'") from exc
        specs = json.loads(data)
        return cls._from_specs(specs)

    @classmethod
    def _from_specs(cls, specs: Iterable[dict[str, str]]) -> "ToolRegistry":
        registry = cls()
        for spec in specs:
            registry.register(
                name=spec["name"],
                category=spec["category"],
                description=spec["description"],
            )
        return registry


DEFAULT_TOOL_REGISTRY = ToolRegistry.with_default_tools()
