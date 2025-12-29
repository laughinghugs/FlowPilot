import json
from pathlib import Path

from agents.manifest import PlanManifestEntry
from agents.pipeline import PipelineAgent
from agents.pipeline_builder import build_pipeline_workspace
from agents.llm import CustomToolDefinition, PlanStep


def write_manifest_entry(path: Path, entry: PlanManifestEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_serializable()) + "\n")


def test_pipeline_agent_executes_plan(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    docs = [
        {"doc_id": "1", "content": "FastAPI enables quick APIs."},
        {"doc_id": "2", "content": "RAG pipelines combine retrieval and generation."},
    ]
    steps = [
        PlanStep(
            tool="InMemoryRetriever",
            rationale="Retrieve relevant docs",
            metadata={"documents": docs, "query": "RAG", "top_k": 1, "output": "retrieved"},
        ),
        PlanStep(
            tool="TemplateLLMGenerator",
            rationale="Draft response from docs",
            metadata={"context_key": "retrieved", "query": "Explain RAG", "output": "response"},
        ),
    ]
    entry = PlanManifestEntry.create(user_message="Explain RAG", steps=steps, system_prompt=None)
    write_manifest_entry(manifest_path, entry)

    agent = PipelineAgent(manifest_path=str(manifest_path))
    result = agent.execute(plan_id=entry.plan_id)

    assert "response" in result.context
    assert "retrieved" in result.context
    assert result.context["retrieved"]["tool"] == "InMemoryRetriever"
    assert result.context["retrieved"]["metadata"]["query"] == "RAG"
    assert result.context["response"]["tool"] == "TemplateLLMGenerator"


def test_build_pipeline_workspace_materializes_files(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    steps = [
        PlanStep(tool="TemplateLLMGenerator", rationale="Respond", metadata={"query": "Hi", "context": []}),
    ]
    custom_tool = CustomToolDefinition(
        name="CustomFetcher",
        purpose="Fetches data",
        inputs={"type": "json"},
        data_sources="API",
        credentials="KEY",
    )
    entry = PlanManifestEntry.create(
        user_message="Say hi",
        steps=steps,
        system_prompt=None,
        custom_tools=[custom_tool],
    )
    write_manifest_entry(manifest_path, entry)

    pipelines_root = tmp_path / "pipelines"
    workspace = build_pipeline_workspace(
        entry.plan_id,
        manifest_path=str(manifest_path),
        output_root=str(pipelines_root),
        run_pipeline=False,
    )

    assert workspace.path.exists()
    assert workspace.pipeline_file.exists()
    assert workspace.custom_tool_files
    assert workspace.codebase_path is not None
    tools_source = (workspace.codebase_path / "tools.py").read_text(encoding="utf-8")
    assert "run_template_llm_generator" in tools_source
    agent_source = (workspace.codebase_path / "agent.py").read_text(encoding="utf-8")
    assert "AgenticPipeline" in agent_source
    tool_payload = json.loads(workspace.custom_tool_files[0].read_text(encoding="utf-8"))
    assert tool_payload["name"] == "CustomFetcher"


def test_build_pipeline_workspace_can_run_pipeline(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    steps = [
        PlanStep(tool="TemplateLLMGenerator", rationale="Respond", metadata={"query": "Hello", "context": []}),
    ]
    entry = PlanManifestEntry.create(user_message="Hello", steps=steps, system_prompt=None)
    write_manifest_entry(manifest_path, entry)

    workspace = build_pipeline_workspace(
        entry.plan_id,
        manifest_path=str(manifest_path),
        output_root=str(tmp_path / "pipelines"),
        run_pipeline=True,
    )

    assert workspace.outputs_file is not None
    outputs = json.loads(workspace.outputs_file.read_text(encoding="utf-8"))
    assert "TemplateLLMGenerator_result" in outputs


def test_pipeline_agent_supports_composite_retriever(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    docs = [
        {"doc_id": "1", "content": "Alpha beta gamma"},
        {"doc_id": "2", "content": "Gamma delta epsilon"},
    ]
    steps = [
        PlanStep(
            tool="CompositeRetriever",
            rationale="Combine multiple retrievers",
            metadata={"documents": docs, "query": "gamma", "top_k": 2, "output": "combined"},
        ),
    ]
    entry = PlanManifestEntry.create(user_message="Test composite", steps=steps, system_prompt=None)
    write_manifest_entry(manifest_path, entry)

    agent = PipelineAgent(manifest_path=str(manifest_path))
    result = agent.execute(plan_id=entry.plan_id)

    payload = result.context["combined"]
    assert payload["tool"] == "CompositeRetriever"
    assert payload["metadata"]["top_k"] == 2


def test_pipeline_agent_handles_custom_tools(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    custom_tool = CustomToolDefinition(
        name="ConfluenceFetcher",
        purpose="Fetches pages from Confluence",
        inputs={"query": "string"},
        data_sources="Confluence API",
        credentials="CONFLUENCE_TOKEN",
        metadata={"owner": "Docs"},
    )
    steps = [
        PlanStep(
            tool="ConfluenceFetcher",
            rationale="Gather source material",
            metadata={"output": "confluence_docs", "filters": {"space": "ED"}},
        )
    ]
    entry = PlanManifestEntry.create(
        user_message="Fetch docs",
        steps=steps,
        system_prompt=None,
        custom_tools=[custom_tool],
    )
    write_manifest_entry(manifest_path, entry)

    agent = PipelineAgent(manifest_path=str(manifest_path))
    result = agent.execute(plan_id=entry.plan_id)

    assert "confluence_docs" in result.context
    payload = result.context["confluence_docs"]
    assert payload["tool"] == "ConfluenceFetcher"
    assert payload["definition_metadata"]["owner"] == "Docs"


def test_pipeline_agent_auto_registers_unknown_tools(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    steps = [
        PlanStep(
            tool="UnknownTool",
            rationale="Future capability",
            metadata={"param": "value", "output": "unknown_output"},
        )
    ]
    entry = PlanManifestEntry.create(user_message="Test unknown tool", steps=steps, system_prompt=None)
    write_manifest_entry(manifest_path, entry)

    agent = PipelineAgent(manifest_path=str(manifest_path))
    result = agent.execute(plan_id=entry.plan_id)

    assert "unknown_output" in result.context
    assert result.context["unknown_output"]["tool"] == "UnknownTool"
    assert result.context["unknown_output"]["status"] == "manifest_defined"
