from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latentmesh import LatentLLM, LatentServe

from examples.common import build_default_flow, seed_sre_memory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve unified LangServe + OpenAI endpoints")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--path", default="/latent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm = LatentLLM(args.model)
    flow = build_default_flow(llm, name="unified-serve", retrieval_k=3)
    seed_sre_memory(flow)

    srv = LatentServe(flow, model_name="latent-prod")
    srv.serve(mode="unified", host=args.host, port=args.port, langserve_path=args.path)


if __name__ == "__main__":
    main()
