from __future__ import annotations

import argparse

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI SDK client against LatentMesh server")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--model", default="latent-prod")
    p.add_argument("--prompt", default="Draft an incident summary for executive stakeholders.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    out = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=260,
        temperature=0.2,
    )
    print(out.choices[0].message.content or "")


if __name__ == "__main__":
    main()

