from __future__ import annotations

import argparse
import json
import urllib.request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible client smoke test")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", default="latent-prod")
    p.add_argument("--prompt", default="Create a post-rollback communication for stakeholders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 220,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as res:
        out = json.loads(res.read().decode("utf-8"))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
