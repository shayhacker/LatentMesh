from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch JSONL client for LatentMesh OpenAI endpoint")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="latent-prod")
    p.add_argument("--in", dest="input_path", required=True, help="Input JSONL with {'prompt': '...'} per line")
    p.add_argument("--out", dest="output_path", required=True, help="Output JSONL path")
    return p.parse_args()


def chat(url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 260,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        out = json.loads(res.read().decode("utf-8"))
    return str(out["choices"][0]["message"]["content"])


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)

    rows: list[dict] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row["prompt"])
        answer = chat(args.url, args.model, prompt)
        rows.append({"prompt": prompt, "response": answer})

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

