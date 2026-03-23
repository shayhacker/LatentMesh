from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentLLM

from examples.common import build_default_flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-query DeepSeek R1 Distill example")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--prompt", default="Write a rollback plan for a failed production migration.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_default_flow(llm, name="deepseek-r1-single", retrieval_k=2)
    print(flow.invoke(args.prompt, max_new_tokens=256, temperature=0.1))


if __name__ == "__main__":
    main()
