from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentGraph, LatentLLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage latent reasoning with DeepSeek R1 Distill")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--prompt", default="Create a 30-minute response plan for elevated 5xx rates.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)

    flow = (
        LatentGraph(llm=llm, name="deepseek-r1-multi", retrieval_k=3)
        .add_stage("triage", transform=lambda x: np.tanh(x))
        .add_stage("policy", transform=lambda x: np.maximum(0.0, x))
        .add_stage("risk", transform=lambda x: np.tanh(x * 0.85))
        .add_stage("write")
        .connect("triage", "policy")
        .connect("policy", "risk")
        .connect("risk", "write")
        .compile(entry_stage="triage", exit_stage="write")
    )

    print(flow.invoke(args.prompt, max_new_tokens=320, temperature=0.15))


if __name__ == "__main__":
    main()
