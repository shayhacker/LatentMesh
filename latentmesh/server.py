"""
FastAPI server for LatentMesh.

Exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint backed by a
single-node LangGraph (``ReasonPrimitive``).
"""

import os
import time
import uuid

import uvicorn
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from latentmesh.core import LatentLLM
from latentmesh.graph import LatentState
from latentmesh.primitives import ReasonPrimitive
from latentmesh.persistent_cache import DiskKVStore, GlobalPrefixCache


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class AppState:
    def __init__(self) -> None:
        self.llm: Optional[LatentLLM] = None
        self.graph: Optional[Any] = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing LatentMesh server…")

    store = DiskKVStore(directory=os.environ.get("LATENTMESH_CACHE_DIR", ".latentmesh_cache"))
    prefix_cache = GlobalPrefixCache(store)

    _state.llm = LatentLLM(
        model_name=os.environ.get("LATENTMESH_MODEL", "HuggingFaceTB/SmolLM-135M"),
        device="cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
        global_cache=prefix_cache,
        debug=os.environ.get("LATENTMESH_DEBUG", "0") == "1",
    )

    builder = StateGraph(LatentState)
    builder.add_node("reasoner", ReasonPrimitive(_state.llm, max_new_tokens=128))
    builder.add_edge(START, "reasoner")
    builder.add_edge("reasoner", END)
    _state.graph = builder.compile()

    print("LatentMesh ready.")
    yield
    print("Shutting down…")


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict:
    if not _state.graph or not _state.llm:
        raise HTTPException(status_code=503, detail="Model backend starting up")

    messages = [m.model_dump() for m in req.messages]
    initial = {"messages": messages, "tokens_so_far": 0}
    final = _state.graph.invoke(initial)

    latent = final.get("latent")
    response_text = latent.text if latent and latent.text else ""

    # Token accounting
    prompt_tokens = latent.input_tokens_uncached + latent.cached_tokens if latent else 0
    completion_tokens = latent.output_tokens if latent else 0
    latent_tokens = final.get("tokens_so_far", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latent_tokens": latent_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
