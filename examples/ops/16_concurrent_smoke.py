from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent smoke test for OpenAI-compatible endpoint")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="latent-prod")
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def send_request(url: str, model: str, index: int) -> tuple[int, float, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Request {index}: give a short incident action list."}],
        "max_tokens": 140,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as res:
        out = json.loads(res.read().decode("utf-8"))
    latency_ms = (time.time() - start) * 1000.0
    text = str(out["choices"][0]["message"]["content"])
    return index, latency_ms, text


def main() -> None:
    args = parse_args()
    latencies: list[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(send_request, args.url, args.model, i) for i in range(args.requests)]
        for fut in as_completed(futs):
            idx, latency_ms, text = fut.result()
            latencies.append(latency_ms)
            print(f"[{idx}] {latency_ms:.1f} ms | {text[:90]}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        print(f"\nsummary: n={len(latencies)} avg_ms={avg:.1f} p95_ms={p95:.1f}")


if __name__ == "__main__":
    main()

