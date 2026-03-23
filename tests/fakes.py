from __future__ import annotations

import sys
import types
from typing import Any


def install_fake_langgraph(monkeypatch) -> None:
    start = "__start__"
    end = "__end__"

    class FakeCompiledGraph:
        def __init__(self, nodes: dict[str, Any], edges: list[tuple[str, str]]) -> None:
            self._nodes = nodes
            self._edges = edges

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            adjacency: dict[str, list[str]] = {}
            for source, target in self._edges:
                adjacency.setdefault(source, []).append(target)

            current = dict(state)
            cursor = start
            while True:
                next_nodes = adjacency.get(cursor, [])
                if not next_nodes:
                    break
                cursor = next_nodes[0]
                if cursor == end:
                    break
                update = self._nodes[cursor](current)
                current.update(update)
            return current

    class FakeStateGraph:
        def __init__(self, _schema: Any) -> None:
            self._nodes: dict[str, Any] = {}
            self._edges: list[tuple[str, str]] = []

        def add_node(self, name: str, fn: Any) -> None:
            self._nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self._edges.append((source, target))

        def compile(self) -> FakeCompiledGraph:
            return FakeCompiledGraph(self._nodes, self._edges)

    mod_root = types.ModuleType("langgraph")
    mod_graph = types.ModuleType("langgraph.graph")
    mod_graph.StateGraph = FakeStateGraph
    mod_graph.START = start
    mod_graph.END = end
    mod_root.graph = mod_graph

    monkeypatch.setitem(sys.modules, "langgraph", mod_root)
    monkeypatch.setitem(sys.modules, "langgraph.graph", mod_graph)


def install_fake_langchain_core(monkeypatch) -> type:
    class FakeRunnableLambda:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, payload):
            return self._fn(payload)

    mod_core = types.ModuleType("langchain_core")
    mod_runnables = types.ModuleType("langchain_core.runnables")
    mod_runnables.RunnableLambda = FakeRunnableLambda
    mod_core.runnables = mod_runnables

    monkeypatch.setitem(sys.modules, "langchain_core", mod_core)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", mod_runnables)
    return FakeRunnableLambda


def install_fake_fastapi(monkeypatch):
    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeFastAPI:
        def __init__(self, title: str) -> None:
            self.title = title
            self.routes: dict[tuple[str, str], Any] = {}

        def get(self, path: str):
            def decorator(fn):
                self.routes[("GET", path)] = fn
                return fn

            return decorator

        def post(self, path: str):
            def decorator(fn):
                self.routes[("POST", path)] = fn
                return fn

            return decorator

    class FakeJSONResponse:
        def __init__(self, content: Any) -> None:
            self.content = content

    mod_fastapi = types.ModuleType("fastapi")
    mod_fastapi.FastAPI = FakeFastAPI
    mod_fastapi.HTTPException = FakeHTTPException

    mod_responses = types.ModuleType("fastapi.responses")
    mod_responses.JSONResponse = FakeJSONResponse

    monkeypatch.setitem(sys.modules, "fastapi", mod_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", mod_responses)

    return FakeFastAPI, FakeHTTPException, FakeJSONResponse


def install_fake_langserve(monkeypatch):
    def add_routes(app, runnable, path: str = "/latent"):
        app.routes[("POST", f"{path}/invoke")] = lambda payload: runnable.invoke(payload)
        app.routes[("POST", f"{path}/batch")] = lambda payload: [runnable.invoke(x) for x in payload]

    mod_langserve = types.ModuleType("langserve")
    mod_langserve.add_routes = add_routes
    monkeypatch.setitem(sys.modules, "langserve", mod_langserve)


def install_fake_uvicorn(monkeypatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def run(app, host: str, port: int) -> None:
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    mod_uvicorn = types.ModuleType("uvicorn")
    mod_uvicorn.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", mod_uvicorn)
    return calls
