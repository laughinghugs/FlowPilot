#!/usr/bin/env python3
"""CLI helper to materialize and execute a manifest-defined pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.pipeline_builder import build_pipeline_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a runnable pipeline workspace for a plan id.")
    parser.add_argument("plan_id", help="Plan ID from plan_manifests.jsonl")
    parser.add_argument(
        "--manifest",
        default="plan_manifests.jsonl",
        help="Path to the plan manifest JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        default="pipelines",
        help="Root directory for pipeline workspaces (default: %(default)s)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Create the workspace without executing the pipeline immediately.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional JSON string with initial context passed to the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_context = {}

    if args.context:
        import json

        try:
            initial_context = json.loads(args.context)
        except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
            raise SystemExit(f"Invalid JSON for --context: {exc}") from exc

    workspace = build_pipeline_workspace(
        args.plan_id,
        manifest_path=args.manifest,
        output_root=args.output_root,
        initial_context=initial_context,
        run_pipeline=not args.no_run,
    )

    print(f"Workspace created at {workspace.path}")  # noqa: T201
    if workspace.outputs_file:
        print(f"Initial pipeline outputs saved to {workspace.outputs_file}")  # noqa: T201


if __name__ == "__main__":
    main()
