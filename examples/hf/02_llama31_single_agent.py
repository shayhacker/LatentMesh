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
    p = argparse.ArgumentParser(description="Single-query Llama example")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt", default="Draft an incident commander update after rollback completion.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_default_flow(llm, name="llama31-single", retrieval_k=2)
    print(flow.invoke(args.prompt, max_new_tokens=256, temperature=0.2))


if __name__ == "__main__":
    main()
