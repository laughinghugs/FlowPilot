from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from agents import PlanningAgent, PipelineAgent
from agents.pipeline_builder import build_pipeline_workspace


class ConversationMessage(TypedDict):
    role: str
    content: str


def _latest_ai_message(history: list[ConversationMessage]) -> str | None:
    for message in reversed(history):
        if message.get("role", "").lower() == "ai":
            content = message.get("content", "").strip()
            if content:
                return content
    return None


def run_planner_chat(system_prompt: str | None) -> None:
    agent = PlanningAgent(system_prompt=system_prompt)
    conversation_history: list[ConversationMessage] = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    print("Chat with the planning agent. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        conversation_history.append({"role": "user", "content": user_input})
        print("Agent is thinking...")

        result = agent.plan(
            user_message=user_input,
            conversation_history=conversation_history,
        )
        conversation_history = list(result.conversation_history)

        if result.clarifying_question:
            print(result.clarifying_question)
            continue

        if result.plan:
            plan_id = result.plan_id or "N/A"
            print(f"Your plan is created and your plan_id is {plan_id}")
            break

        ai_message = _latest_ai_message(conversation_history)
        if ai_message:
            print(f"AI: {ai_message}")
        else:
            print("Planner did not return a plan or clarification.")


def run_pipeline_chat(
    plan_id: str,
    *,
    manifest_path: str,
    pipelines_root: str,
) -> None:
    workspace = build_pipeline_workspace(
        plan_id,
        manifest_path=manifest_path,
        output_root=pipelines_root,
        run_pipeline=False,
    )
    agent = PipelineAgent(manifest_path=manifest_path)
    history_path = workspace.path / "pipeline_chat.jsonl"

    print(f"Connected to pipeline {plan_id}. Type 'exit' to leave.\n")

    while True:
        try:
            user_input = input("You (pipeline): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting pipeline chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Logging out of pipeline.")
            break

        context = {"query": user_input}
        result = agent.execute(plan_id=plan_id, context=context)
        response = _extract_pipeline_response(result.context)
        print(f"Pipeline: {response}")
        _append_pipeline_history(history_path, user_input, response)


def _extract_pipeline_response(context: dict[str, object]) -> str:
    for key in ("llm_response", "response", "output", "result"):
        value = context.get(key)
        if isinstance(value, str):
            return value
    for value in context.values():
        if isinstance(value, str):
            return value
    return json.dumps(context, indent=2, default=str)


def _append_pipeline_history(path: Path, user_message: str, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": "user", "content": user_message}) + "\n")
        handle.write(json.dumps({"role": "pipeline", "content": response}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the planning agent or an existing pipeline.")
    parser.add_argument(
        "--mode",
        choices=("planner", "pipeline"),
        default="planner",
        help="Choose 'planner' to design plans or 'pipeline' to chat with an existing plan.",
    )
    parser.add_argument(
        "--system",
        dest="system_prompt",
        help="Optional system prompt to steer the planner.",
    )
    parser.add_argument(
        "--plan-id",
        help="Plan id to connect to in pipeline mode. If omitted you will be prompted.",
    )
    parser.add_argument(
        "--manifest",
        default="plan_manifests.jsonl",
        help="Path to the manifest file (default: %(default)s).",
    )
    parser.add_argument(
        "--pipelines-root",
        default="pipelines",
        help="Root directory for generated pipeline workspaces (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.mode == "planner":
        run_planner_chat(args.system_prompt)
        return

    plan_id = args.plan_id or input("Enter plan id: ").strip()
    if not plan_id:
        raise SystemExit("Plan id is required for pipeline mode.")

    run_pipeline_chat(plan_id, manifest_path=args.manifest, pipelines_root=args.pipelines_root)


if __name__ == "__main__":
    main()
