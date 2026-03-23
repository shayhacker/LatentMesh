from __future__ import annotations

import numpy as np

from latentmesh import LatentLLM, LatentGraph

from .fakes import install_fake_langchain_core


def build_workflow():
    llm = LatentLLM("mock://dev", backend="mock")
    return (
        LatentGraph(llm=llm)
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("writer")
        .connect("planner", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )


def test_runnable_fallback_invoke_with_dict() -> None:
    workflow = build_workflow()
    runnable = workflow.as_runnable()
    out = runnable.invoke({"question": "How do I rollback?", "max_new_tokens": 48})
    assert isinstance(out, str)
    assert out


def test_runnable_fallback_batch() -> None:
    workflow = build_workflow()
    runnable = workflow.as_runnable()
    out = runnable.batch(["Q1", {"question": "Q2"}])
    assert len(out) == 2


def test_runnable_uses_langchain_runnablelambda_when_available(monkeypatch) -> None:
    fake_runnable_type = install_fake_langchain_core(monkeypatch)
    workflow = build_workflow()
    runnable = workflow.as_runnable()
    assert isinstance(runnable, fake_runnable_type)
    assert isinstance(runnable.invoke({"question": "Q"}), str)
