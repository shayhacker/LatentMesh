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


class RagState(TypedDict, total=False):
    question: str
    context: str
    answer: str


DOCS = [
    "Rollback policy: revert within five minutes if checkout p95 is above 700ms for two windows.",
    "Database safety: freeze writes before schema rollback and validate replica lag before re-enable.",
    "API mitigation: shed non-critical traffic before scaling up worker pools.",
    "Comms: post internal incident update every 15 minutes with action items and owner names.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph RAG + LatentGraph answer generation")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--prompt", default="Checkout latency doubled after deploy. Provide remediation steps.")
    p.add_argument("--k", type=int, default=2)
    return p.parse_args()


def topk_docs(llm: LatentLLM, query: str, k: int) -> list[str]:
    query_lat = llm.embed(query)
    scores: list[tuple[float, str]] = []
    for doc in DOCS:
        doc_lat = llm.embed(doc)
        denom = float(np.linalg.norm(query_lat) * np.linalg.norm(doc_lat)) + 1e-9
        score = float(np.dot(query_lat, doc_lat) / denom)
        scores.append((score, doc))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scores[:k]]


def build_latent_flow(llm: LatentLLM):
    flow = (
        LatentGraph(llm=llm, name="rag-latent", retrieval_k=2)
        .add_stage("ground", transform=lambda x: np.tanh(x))
        .add_stage("policy", transform=lambda x: np.maximum(0.0, x))
        .add_stage("write")
        .connect("ground", "policy")
        .connect("policy", "write")
        .compile(entry_stage="ground", exit_stage="write")
    )
    flow.add_example(
        "How should updates be sent during incidents?",
        "Use regular updates with clear owner + ETA and explicit next action.",
    )
    return flow


def build_graph(llm: LatentLLM, flow, k: int):
    graph = StateGraph(RagState)

    def retrieve_node(state: RagState) -> RagState:
        docs = topk_docs(llm, state["question"], k)
        return {"context": "\n".join(f"- {doc}" for doc in docs)}

    def latent_node(state: RagState) -> RagState:
        prompt = (
            "Use the context for factual grounding.\n"
            f"Context:\n{state['context']}\n\n"
            f"Question: {state['question']}"
        )
        answer = flow.invoke(prompt, max_new_tokens=320, temperature=0.1)
        return {"answer": answer}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("latent_answer", latent_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "latent_answer")
    graph.add_edge("latent_answer", END)
    return graph.compile()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_latent_flow(llm)
    app = build_graph(llm, flow, k=args.k)
    out = app.invoke({"question": args.prompt})
    print("Context:\n", out["context"])
    print("\nAnswer:\n", out["answer"])


if __name__ == "__main__":
    main()

