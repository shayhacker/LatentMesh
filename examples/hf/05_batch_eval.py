from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentLLM

from examples.common import build_default_flow, seed_sre_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluation with one GPU model")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_default_flow(llm, name="batch-eval", retrieval_k=3)
    seed_sre_memory(flow)

    prompts = [
        "Primary DB failover raised replication lag. Build an immediate response plan.",
        "Cache cluster eviction storm started after deploy. How do we stabilize quickly?",
        "API p95 doubled after feature flag rollout. Give a rollback + verification checklist.",
    ]

    outs = flow.batch(prompts, max_new_tokens=220, temperature=0.2)
    for i, out in enumerate(outs, start=1):
        print(f"\n--- RESULT {i} ---\n{out}")


if __name__ == "__main__":
    main()
