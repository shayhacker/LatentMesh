"""LangGraph + Ollama example with latent stages.

Install:
  pip install -e '.[ollama,langgraph]'

Requirements:
  - Ollama running locally
  - ollama pull nomic-embed-text
  - ollama pull llama3.1:8b
"""

from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph


def build_workflow():
    llm = LatentLLM("ollama:llama3.1:8b", backend="ollama", embedding_model="nomic-embed-text")

    graph = (
        LatentGraph(llm=llm, name="ollama-latent")
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
        .add_stage("writer")
        .connect("planner", "critic")
        .connect("critic", "writer")
    )
    return graph.compile(entry_stage="planner", exit_stage="writer")


def main() -> None:
    workflow = build_workflow()
    print(workflow.invoke("Give me a rollback checklist for a failed production deploy."))


if __name__ == "__main__":
    main()
