"""Code generation helpers for manifest-defined pipelines."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Protocol

try:  # pragma: no cover - optional
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None  # type: ignore[misc]
    OpenAI = None  # type: ignore[misc]

from .manifest import PlanManifestEntry

LLM_CODE_PROMPT = """You are an elite Python engineer. Generate production-quality code for the tools and agent
pipeline described in the manifest JSON. Return valid JSON with keys "tools.py", "pipeline.py", and "agent.py".

Requirements:
- tools.py: define one function per tool (run_<slug>(query: str, context: dict, metadata: dict) -> dict) with docstrings
  explaining behavior. Implement deterministic placeholder logic using the metadata (no external imports beyond stdlib).
- pipeline.py: describe PIPELINE_STEPS referencing the tool functions. Provide run_pipeline(query, context=None) that
  executes steps sequentially, storing outputs using metadata["output"] or "<tool>_result".
- agent.py: define AgenticPipeline with run(self, query, context=None) that delegates to pipeline.run_pipeline.

Ensure the code is syntactically correct and self-contained."""


class PipelineCodeGenerator(Protocol):
    """Protocol for code generators that turn a manifest entry into python modules."""

    def generate(self, entry: PlanManifestEntry) -> dict[str, str]:
        ...


class TemplatePipelineCodeGenerator(PipelineCodeGenerator):
    """Deterministic fallback generator used in tests/offline environments."""

    def generate(self, entry: PlanManifestEntry) -> dict[str, str]:
        return {
            "__init__.py": self._render_init(),
            "tools.py": self._render_tools(entry),
            "pipeline.py": self._render_pipeline(entry),
            "agent.py": self._render_agent(entry),
        }

    @staticmethod
    def _render_init() -> str:
        return textwrap.dedent(
            '''\
            """Auto-generated agent pipeline package."""
            from .agent import AgenticPipeline

            __all__ = ["AgenticPipeline"]
            '''
        )

    def _render_tools(self, entry: PlanManifestEntry) -> str:
        lines = [
            f'"""Auto-generated tool implementations for plan {entry.plan_id}."""',
            "from __future__ import annotations",
            "",
            "from typing import Any, Dict",
            "",
        ]
        seen: set[str] = set()
        for step in entry.steps:
            tool = step.tool
            if tool in seen:
                continue
            seen.add(tool)
            func_name = f"run_{_slugify(tool)}"
            doc = textwrap.dedent(
                f"""\
                Tool: {tool}
                Rationale: {step.rationale}
                Expected metadata keys: {sorted(step.metadata.keys()) if step.metadata else []}
                """
            ).strip()
            body = textwrap.dedent(
                f"""\
                def {func_name}(query: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
                    \"\"\"{doc}\"\"\"
                    return {{
                        "tool": "{tool}",
                        "query": query,
                        "metadata": metadata,
                        "context_snapshot": list(context.keys()),
                        "status": "pending_implementation",
                    }}
                """
            )
            lines.append(body)

        if len(seen) == 0:
            lines.append(
                textwrap.dedent(
                    """\
                    def run_placeholder(query: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
                        \"\"\"Placeholder when no tools are defined.\"\"\"
                        return {"tool": "placeholder", "query": query, "metadata": metadata}
                    """
                )
            )

        return "\n".join(lines)

    def _render_pipeline(self, entry: PlanManifestEntry) -> str:
        step_blocks = []
        for step in entry.steps:
            runner = f"tool_impl.run_{_slugify(step.tool)}"
            metadata_json = json.dumps(step.metadata or {}, indent=4)
            block = textwrap.dedent(
                f"""\
                {{
                    "name": "{step.tool}",
                    "rationale": {step.rationale!r},
                    "metadata": {metadata_json},
                    "runner": {runner},
                }}"""
            )
            step_blocks.append(block)

        steps_literal = ",\n    ".join(step_blocks) if step_blocks else ""
        return textwrap.dedent(
            f"""\
            \"\"\"Auto-generated pipeline orchestration for plan {entry.plan_id}.\"\"\"
            from __future__ import annotations

            from typing import Any, Dict, List

            from . import tools as tool_impl

            PIPELINE_STEPS: List[dict[str, Any]] = [
                {steps_literal}
            ]


            def run_pipeline(query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
                \"\"\"Execute the manifest-defined steps sequentially.\"\"\"
                ctx = dict(context or {{}})
                ctx.setdefault("user_query", query)
                for step in PIPELINE_STEPS:
                    runner = step["runner"]
                    metadata = dict(step.get("metadata") or {{}})
                    result = runner(query, ctx, metadata)
                    output_key = metadata.get("output") or f"{{step['name']}}_result"
                    ctx[output_key] = result
                return ctx
            """
        )

    @staticmethod
    def _render_agent(entry: PlanManifestEntry) -> str:
        return textwrap.dedent(
            """\
            \"\"\"Agent façade for executing the generated pipeline.\"\"\"
            from __future__ import annotations

            from typing import Any, Dict

            from .pipeline import run_pipeline


            class AgenticPipeline:
                \"\"\"Simple agent that runs the pipeline for a user query.\"\"\"

                def run(self, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
                    return run_pipeline(query, context=context)
            """
        )


class OpenAIPipelineCodeGenerator(PipelineCodeGenerator):
    """LLM-backed generator that produces concrete tool + agent code."""

    def __init__(self, client: OpenAI, model: str | None = None) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError("openai package is required for OpenAIPipelineCodeGenerator")
        self._client = client
        self._model = model or os.getenv("OPENAI_CODE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.1-mini"

    def generate(self, entry: PlanManifestEntry) -> dict[str, str]:  # noqa: D401
        manifest_json = json.dumps(entry.to_serializable(), indent=2)
        messages = [
            {"role": "system", "content": LLM_CODE_PROMPT},
            {
                "role": "user",
                "content": f"Manifest JSON:\\n{manifest_json}",
            },
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return {name: data[name] for name in ("__init__.py", "tools.py", "pipeline.py", "agent.py") if name in data}


class AzureOpenAIPipelineCodeGenerator(PipelineCodeGenerator):
    """LLM-backed generator for Azure OpenAI deployments."""

    def __init__(self, client: AzureOpenAI, deployment: str) -> None:
        if AzureOpenAI is None:  # pragma: no cover
            raise ImportError("openai package with Azure support is required for AzureOpenAIPipelineCodeGenerator")
        self._client = client
        self._deployment = deployment

    def generate(self, entry: PlanManifestEntry) -> dict[str, str]:  # noqa: D401
        manifest_json = json.dumps(entry.to_serializable(), indent=2)
        messages = [
            {"role": "system", "content": LLM_CODE_PROMPT},
            {"role": "user", "content": f"Manifest JSON:\\n{manifest_json}"},
        ]
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return {name: data[name] for name in ("__init__.py", "tools.py", "pipeline.py", "agent.py") if name in data}


def build_code_generator_from_env() -> PipelineCodeGenerator:
    """Factory that prefers LLM-backed generators and falls back to the template generator."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    try:
        if provider == "openai" and OpenAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAIPipelineCodeGenerator(OpenAI(api_key=api_key))
        elif provider == "azure_openai" and AzureOpenAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if api_key and endpoint and deployment:
                client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
                return AzureOpenAIPipelineCodeGenerator(client, deployment)
    except Exception:  # pragma: no cover - fallback path
        pass
    return TemplatePipelineCodeGenerator()


def _slugify(value: str) -> str:
    import re

    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "tool"


__all__ = [
    "PipelineCodeGenerator",
    "TemplatePipelineCodeGenerator",
    "OpenAIPipelineCodeGenerator",
    "AzureOpenAIPipelineCodeGenerator",
    "build_code_generator_from_env",
]
