"""LangServe serving for a latent LangGraph workflow.

Install:
  pip install -e '.[langgraph,serve]'

Run:
  python -m latentmesh.examples.langserve_server
"""

from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph, LatentServe


def build_workflow():
    llm = LatentLLM("mock://server", backend="mock")
    graph = (
        LatentGraph(llm=llm, name="langserve-latent")
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
        .add_stage("writer")
        .connect("planner", "critic")
        .connect("critic", "writer")
    )
    return graph.compile(entry_stage="planner", exit_stage="writer")


def main() -> None:
    workflow = build_workflow()
    gateway = LatentServe(workflow, model_name="latent-langgraph")
    gateway.serve(mode="langserve", host="0.0.0.0", port=8000, langserve_path="/latent")


if __name__ == "__main__":
    main()
