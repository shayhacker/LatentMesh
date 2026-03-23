from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentGraph, LatentLLM

from examples.common import print_trace, seed_sre_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ollama multi-stage + retrieval example")
    p.add_argument("--model", default="ollama:llama3.1:8b")
    p.add_argument("--embed", default="nomic-embed-text")
    p.add_argument("--prompt", default="Alert storm after deploy. Build a response timeline for the next hour.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model, backend="ollama", embedding_model=args.embed)

    flow = (
        LatentGraph(llm=llm, name="ollama-multi", retrieval_k=3)
        .add_stage("triage", transform=lambda x: np.tanh(x))
        .add_stage("cost", transform=lambda x: np.maximum(0.0, x))
        .add_stage("policy", transform=lambda x: np.tanh(x * 0.75))
        .add_stage("write")
        .connect("triage", "cost")
        .connect("cost", "policy")
        .connect("policy", "write")
        .compile(entry_stage="triage", exit_stage="write")
    )
    seed_sre_memory(flow)

    ans, trace = flow.run(args.prompt, max_new_tokens=320, temperature=0.15)
    print(ans)
    print_trace(trace)


if __name__ == "__main__":
    main()
