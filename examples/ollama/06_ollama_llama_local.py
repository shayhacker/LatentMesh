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
    p = argparse.ArgumentParser(description="Local Ollama single-GPU workflow")
    p.add_argument("--model", default="ollama:llama3.1:8b")
    p.add_argument("--embed", default="nomic-embed-text")
    p.add_argument("--prompt", default="Generate a safe incident communication for internal engineering teams.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model, backend="ollama", embedding_model=args.embed)
    flow = build_default_flow(llm, name="ollama-local", retrieval_k=2)
    print(flow.invoke(args.prompt, max_new_tokens=240, temperature=0.2))


if __name__ == "__main__":
    main()
