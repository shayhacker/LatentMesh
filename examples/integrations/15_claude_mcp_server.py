from __future__ import annotations

import json
import urllib.request

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("latentmesh")


def _chat_openai_compat(
    *,
    prompt: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        out = json.loads(res.read().decode("utf-8"))
    return str(out["choices"][0]["message"]["content"])


@mcp.tool()
def latent_chat(
    prompt: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str = "latent-prod",
    max_tokens: int = 320,
    temperature: float = 0.2,
) -> str:
    """Send prompt to a local LatentMesh OpenAI-compatible endpoint."""
    return _chat_openai_compat(
        prompt=prompt,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    mcp.run()

