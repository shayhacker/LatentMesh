"""Quickstart: OOP latent workflow compiled via LangGraph (fallback if missing)."""

from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph


def build_workflow():
    llm = LatentLLM("mock://dev", backend="mock")

    graph = (
        LatentGraph(llm=llm, name="quickstart", retrieval_k=2)
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
        .add_stage("writer")
        .connect("planner", "critic")
        .connect("critic", "writer")
        .add_example("How to rollback a bad deploy?", "Rollback to the last healthy revision and re-run smoke tests.")
    )
    return graph.compile(entry_stage="planner", exit_stage="writer")


def main() -> None:
    workflow = build_workflow()
    answer, trace = workflow.run("Deployment failed after migration timeout. What should I do first?")
    print(answer)
    print("uses_langgraph=", workflow.uses_langgraph)
    print("stages=", [step.stage for step in trace.steps])


if __name__ == "__main__":
    main()
