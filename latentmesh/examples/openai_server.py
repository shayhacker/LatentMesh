"""OpenAI-compatible serving for a latent LangGraph workflow.

Install:
  pip install -e '.[serve]'

Run:
  python -m latentmesh.examples.openai_server
"""

from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph, LatentServe


def build_workflow():
    llm = LatentLLM("mock://server", backend="mock")
    graph = (
        LatentGraph(llm=llm, name="openai-compatible")
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("writer")
        .connect("planner", "writer")
    )
    workflow = graph.compile(entry_stage="planner", exit_stage="writer")
    workflow.add_example("How to rollback a deploy?", "Rollback to the last healthy build and validate core endpoints.")
    return workflow


def main() -> None:
    workflow = build_workflow()
    gateway = LatentServe(workflow, model_name="latent-langgraph")
    gateway.serve(mode="openai", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
