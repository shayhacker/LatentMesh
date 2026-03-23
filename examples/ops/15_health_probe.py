from __future__ import annotations

import argparse
import json
import urllib.request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Health/model probe for a running LatentMesh server")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    return p.parse_args()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as res:
        return json.loads(res.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    health = fetch_json(f"{base}/healthz")
    models = fetch_json(f"{base}/v1/models")
    print("health:", health)
    print("models:", json.dumps(models, indent=2))


if __name__ == "__main__":
    main()

