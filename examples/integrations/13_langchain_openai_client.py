from __future__ import annotations

import argparse
from typing import Sequence

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangChain client against LatentMesh OpenAI endpoint")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--model", default="latent-prod")
    p.add_argument("--prompt", default="Give a structured rollback playbook for a failed release.")
    return p.parse_args()


def invoke_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )
    msg: Sequence[HumanMessage] = [HumanMessage(content=prompt)]
    out = chat.invoke(msg)
    return str(out.content)


def main() -> None:
    args = parse_args()
    print(
        invoke_chat(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=args.prompt,
        )
    )


if __name__ == "__main__":
    main()

