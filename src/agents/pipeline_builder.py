"""Helpers for materializing manifest-defined plans into runnable pipeline workspaces."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .manifest import PlanManifestReader
from .pipeline import PipelineAgent


@dataclass(frozen=True)
class PipelineWorkspace:
    """Describes the files generated for a pipeline plan."""

    plan_id: str
    path: Path
    pipeline_file: Path
    custom_tool_files: tuple[Path, ...]
    outputs_file: Path | None = None


def build_pipeline_workspace(
    plan_id: str,
    *,
    manifest_path: str = "plan_manifests.jsonl",
    output_root: str = "pipelines",
    initial_context: dict[str, Any] | None = None,
    run_pipeline: bool = True,
) -> PipelineWorkspace:
    """
    Create a filesystem workspace for the given plan id and optionally execute it once.

    - Writes the manifest snapshot to ``pipelines/<plan_id>/pipeline.json``.
    - Materializes custom tool definitions into ``custom_tools/<slug>.json`` files.
    - Executes the pipeline (unless ``run_pipeline`` is False) and saves outputs as JSON.
    """

    reader = PlanManifestReader(manifest_path)
    entry = reader.get(plan_id)

    workspace_dir = Path(output_root).joinpath(plan_id)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    pipeline_file = workspace_dir / "pipeline.json"
    pipeline_file.write_text(
        json.dumps(entry.to_serializable(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    custom_tool_files: list[Path] = []
    if entry.custom_tools:
        tools_dir = workspace_dir / "custom_tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        for tool in entry.custom_tools:
            file_name = f"{_slugify(tool.name)}.json"
            tool_path = tools_dir / file_name
            tool_path.write_text(json.dumps(_jsonify(tool), indent=2, ensure_ascii=False), encoding="utf-8")
            custom_tool_files.append(tool_path)

    outputs_file: Path | None = None
    if run_pipeline:
        agent = PipelineAgent(manifest_path=manifest_path)
        result = agent.execute(plan_id=plan_id, context=initial_context or {})
        outputs = workspace_dir / "outputs.json"
        outputs.write_text(json.dumps(_jsonify(result.context), indent=2, ensure_ascii=False), encoding="utf-8")
        outputs_file = outputs

    return PipelineWorkspace(
        plan_id=plan_id,
        path=workspace_dir,
        pipeline_file=pipeline_file,
        custom_tool_files=tuple(custom_tool_files),
        outputs_file=outputs_file,
    )


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "custom-tool"


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Iterable):
        return [_jsonify(item) for item in value]
    return repr(value)


__all__ = ["PipelineWorkspace", "build_pipeline_workspace"]
