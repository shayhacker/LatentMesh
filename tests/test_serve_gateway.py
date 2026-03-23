from __future__ import annotations

import numpy as np
import pytest

from latentmesh import LatentLLM, LatentGraph, LatentServe

from .fakes import (
    install_fake_fastapi,
    install_fake_langchain_core,
    install_fake_langserve,
    install_fake_uvicorn,
)


def build_workflow():
    llm = LatentLLM("mock://server", backend="mock")
    return (
        LatentGraph(llm=llm)
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("writer")
        .connect("planner", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )


def test_chat_completions_shape() -> None:
    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    response = gateway.chat_completions(
        {
            "model": "latent-langgraph",
            "messages": [{"role": "user", "content": "Rollback checklist"}],
            "max_tokens": 32,
        }
    )
    assert response["object"] == "chat.completion"
    assert response["model"] == "latent-langgraph"
    assert response["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_rejects_empty_messages() -> None:
    gateway = LatentServe(build_workflow())
    with pytest.raises(ValueError, match="non-empty"):
        gateway.chat_completions({"messages": []})


def test_models_shape() -> None:
    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    models = gateway.models()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "latent-langgraph"


def test_create_openai_app_registers_routes(monkeypatch) -> None:
    install_fake_fastapi(monkeypatch)
    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    app = gateway.create_openai_app()

    assert ("GET", "/v1/models") in app.routes
    assert ("POST", "/v1/chat/completions") in app.routes
    assert ("GET", "/healthz") in app.routes


def test_openai_route_returns_http_400(monkeypatch) -> None:
    _, fake_http_exception, _ = install_fake_fastapi(monkeypatch)
    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    app = gateway.create_openai_app()
    chat = app.routes[("POST", "/v1/chat/completions")]

    with pytest.raises(fake_http_exception) as exc:
        chat({"messages": []})

    assert exc.value.status_code == 400


def test_create_langserve_app(monkeypatch) -> None:
    install_fake_fastapi(monkeypatch)
    install_fake_langchain_core(monkeypatch)
    install_fake_langserve(monkeypatch)

    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    app = gateway.create_langserve_app(path="/latent")

    assert ("POST", "/latent/invoke") in app.routes
    assert ("POST", "/latent/batch") in app.routes


def test_create_unified_app(monkeypatch) -> None:
    install_fake_fastapi(monkeypatch)
    install_fake_langchain_core(monkeypatch)
    install_fake_langserve(monkeypatch)

    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    app = gateway.create_unified_app(langserve_path="/latent")

    assert ("POST", "/latent/invoke") in app.routes
    assert ("POST", "/v1/chat/completions") in app.routes


def test_serve_runs_uvicorn_for_openai(monkeypatch) -> None:
    install_fake_fastapi(monkeypatch)
    calls = install_fake_uvicorn(monkeypatch)

    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    gateway.serve(mode="openai", host="127.0.0.1", port=9000)

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9000


def test_serve_rejects_invalid_mode(monkeypatch) -> None:
    calls = install_fake_uvicorn(monkeypatch)
    del calls

    gateway = LatentServe(build_workflow(), model_name="latent-langgraph")
    with pytest.raises(ValueError, match="mode"):
        gateway.serve(mode="invalid")
