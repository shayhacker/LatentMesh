"""LangGraph + HuggingFace example with latent stages.

Install:
  pip install -e '.[transformers,langgraph]'
"""

from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph


def build_workflow():
    llm = LatentLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    graph = (
        LatentGraph(llm=llm, name="hf-latent")
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("writer")
        .connect("planner", "writer")
    )
    return graph.compile(entry_stage="planner", exit_stage="writer")


def main() -> None:
    workflow = build_workflow()
    print(workflow.invoke("Write a concise SQL query for duplicate user emails."))


if __name__ == "__main__":
    main()
