from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentLLM

from examples.common import build_default_flow, print_trace, seed_sre_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seeded memory support workflow")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt", default="DB migration timed out and checkout flow is failing. What now?")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_default_flow(llm, name="memory-seeded", retrieval_k=3)
    seed_sre_memory(flow)

    ans, trace = flow.run(args.prompt, max_new_tokens=320, temperature=0.1)
    print(ans)
    print_trace(trace)


if __name__ == "__main__":
    main()
