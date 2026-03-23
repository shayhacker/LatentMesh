"""Serving gateway for latent workflows (LangServe + OpenAI-compatible API)."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any
from uuid import uuid4

from .workflow import LatentWorkflow


@dataclass(slots=True)
class LatentServe:
    """Unified serving gateway for a latent LangGraph workflow."""

    workflow: LatentWorkflow
    model_name: str = "latent-langgraph"

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("payload.messages must be a non-empty list")

        prompt = _messages_to_prompt(messages)
        max_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.2))

        answer = self.workflow.invoke(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(answer.split()))

        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time()),
            "model": str(payload.get("model", self.model_name)),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "latentmesh",
                }
            ],
        }

    def create_langserve_app(self, *, path: str = "/latent"):
        """Create FastAPI app with LangServe routes."""

        try:
            from fastapi import FastAPI
            from langserve import add_routes
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LangServe serving requires 'fastapi', 'langchain-core', and 'langserve'. Install extras: .[langgraph,serve]"
            ) from exc

        app = FastAPI(title=f"{self.model_name} LangServe API")
        add_routes(app, self.workflow.as_runnable(), path=path)

        @app.get("/healthz")
        def healthz() -> dict[str, str]:
            return {"status": "ok"}

        return app

    def create_openai_app(self):
        """Create FastAPI app exposing OpenAI-compatible routes."""

        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenAI-compatible serving requires 'fastapi'. Install extras: .[serve]") from exc

        app = FastAPI(title=f"{self.model_name} OpenAI API")

        @app.get("/v1/models")
        def list_models():
            return JSONResponse(self.models())

        @app.post("/v1/chat/completions")
        def chat_completions(payload: dict[str, Any]):
            try:
                return JSONResponse(self.chat_completions(payload))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app.get("/healthz")
        def healthz() -> dict[str, str]:
            return {"status": "ok"}

        return app

    def create_unified_app(self, *, langserve_path: str = "/latent"):
        """Create one FastAPI app exposing LangServe and OpenAI-compatible endpoints."""

        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            from langserve import add_routes
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unified serving requires 'fastapi', 'langchain-core', and 'langserve'. Install extras: .[langgraph,serve]"
            ) from exc

        app = FastAPI(title=f"{self.model_name} Gateway")
        add_routes(app, self.workflow.as_runnable(), path=langserve_path)

        @app.get("/v1/models")
        def list_models():
            return JSONResponse(self.models())

        @app.post("/v1/chat/completions")
        def chat_completions(payload: dict[str, Any]):
            try:
                return JSONResponse(self.chat_completions(payload))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app.get("/healthz")
        def healthz() -> dict[str, str]:
            return {"status": "ok"}

        return app

    def serve(
        self,
        *,
        mode: str = "unified",
        host: str = "0.0.0.0",
        port: int = 8000,
        langserve_path: str = "/latent",
    ) -> None:
        """Run uvicorn for the selected API mode."""

        try:
            import uvicorn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Serving requires 'uvicorn'. Install extras: .[serve]") from exc

        normalized = mode.lower().strip()
        if normalized == "openai":
            app = self.create_openai_app()
        elif normalized == "langserve":
            app = self.create_langserve_app(path=langserve_path)
        elif normalized == "unified":
            app = self.create_unified_app(langserve_path=langserve_path)
        else:
            raise ValueError("mode must be one of: openai, langserve, unified")

        uvicorn.run(app, host=host, port=port)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _message_content_to_text(message.get("content", ""))
        lines.append(f"[{role}] {content}")
    return "\n".join(lines).strip()


__all__ = ["LatentServe"]
