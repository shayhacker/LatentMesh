"""Simple benchmark harness for LatentMesh workflows.

Example:
  python -m latentmesh.examples.benchmark_runner --requests 1000 --warmup 100
"""

from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import numpy as np

from latentmesh import LatentLLM, LatentGraph


def build_workflow():
    llm = LatentLLM("mock://bench", backend="mock")
    return (
        LatentGraph(llm=llm, name="benchmark", retrieval_k=2)
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
        .add_stage("writer")
        .connect("planner", "critic")
        .connect("critic", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * p)
    return float(sorted_values[idx])


def run_benchmark(requests: int, warmup: int, max_new_tokens: int) -> dict[str, float | int]:
    workflow = build_workflow()

    for i in range(max(0, warmup)):
        workflow.invoke(f"warmup {i}", max_new_tokens=max_new_tokens)

    latencies_ms: list[float] = []
    start = time.perf_counter()

    for i in range(max(1, requests)):
        t0 = time.perf_counter()
        workflow.invoke(f"request {i}", max_new_tokens=max_new_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_s = time.perf_counter() - start
    return {
        "requests": int(requests),
        "warmup": int(warmup),
        "max_new_tokens": int(max_new_tokens),
        "throughput_rps": round(requests / max(total_s, 1e-9), 2),
        "latency_mean_ms": round(mean(latencies_ms), 3),
        "latency_p50_ms": round(percentile(latencies_ms, 0.50), 3),
        "latency_p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "latency_p99_ms": round(percentile(latencies_ms, 0.99), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LatentMesh workflow runtime")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        requests=max(1, args.requests),
        warmup=max(0, args.warmup),
        max_new_tokens=max(1, args.max_new_tokens),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
