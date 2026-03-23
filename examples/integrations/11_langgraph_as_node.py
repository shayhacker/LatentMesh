from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentGraph, LatentLLM


class AgentState(TypedDict, total=False):
    question: str
    severity: str
    answer: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use LatentGraph as a node in an existing LangGraph app")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt", default="Payments API is flapping at 40% 5xx. Create an on-call action plan.")
    return p.parse_args()


def build_latent_flow(llm: LatentLLM):
    return (
        LatentGraph(llm=llm, name="embedded-latent", retrieval_k=2)
        .add_stage("plan", transform=lambda x: np.tanh(x))
        .add_stage("risk", transform=lambda x: np.maximum(0.0, x))
        .add_stage("write")
        .connect("plan", "risk")
        .connect("risk", "write")
        .compile(entry_stage="plan", exit_stage="write")
    )


def build_graph(latent_flow):
    graph = StateGraph(AgentState)

    def route_node(state: AgentState) -> AgentState:
        text = state["question"].lower()
        sev = "critical" if any(x in text for x in ("5xx", "outage", "down", "incident")) else "normal"
        return {"severity": sev}

    def latent_node(state: AgentState) -> AgentState:
        prompt = f"[severity:{state['severity']}] {state['question']}"
        answer = latent_flow.invoke(prompt, max_new_tokens=300, temperature=0.15)
        return {"answer": answer}

    graph.add_node("route", route_node)
    graph.add_node("latent", latent_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "latent")
    graph.add_edge("latent", END)
    return graph.compile()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    latent_flow = build_latent_flow(llm)
    app = build_graph(latent_flow)
    out = app.invoke({"question": args.prompt})
    print(out["answer"])


if __name__ == "__main__":
    main()

