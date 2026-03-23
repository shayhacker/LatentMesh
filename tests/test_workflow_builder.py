from __future__ import annotations

import numpy as np
import pytest

from latentmesh import LatentLLM, LatentGraph, LinearTransform

from .fakes import install_fake_langgraph


def test_builder_rejects_duplicate_stage() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    builder = LatentGraph(llm=llm)
    builder.add_stage("planner")
    with pytest.raises(ValueError, match="already exists"):
        builder.add_stage("planner")


def test_builder_rejects_unknown_edge_nodes() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    builder = LatentGraph(llm=llm).add_stage("planner")
    with pytest.raises(ValueError, match="unknown target"):
        builder.connect("planner", "writer")


def test_builder_requires_stages_before_compile() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    builder = LatentGraph(llm=llm)
    with pytest.raises(ValueError, match="at least one stage"):
        builder.compile(entry_stage="planner", exit_stage="writer")


def test_compile_rejects_cycle() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    builder = (
        LatentGraph(llm=llm)
        .add_stage("a")
        .add_stage("b")
        .connect("a", "b")
        .connect("b", "a")
    )
    with pytest.raises(ValueError, match="DAG"):
        builder.compile(entry_stage="a", exit_stage="b")


def test_compile_uses_fallback_without_langgraph() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    workflow = (
        LatentGraph(llm=llm)
        .add_stage("planner")
        .add_stage("writer")
        .connect("planner", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )
    assert workflow.uses_langgraph is False


def test_compile_uses_langgraph_when_available(monkeypatch) -> None:
    install_fake_langgraph(monkeypatch)

    llm = LatentLLM("mock://dev", backend="mock")
    workflow = (
        LatentGraph(llm=llm)
        .add_stage("planner")
        .add_stage("writer")
        .connect("planner", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )
    assert workflow.uses_langgraph is True


def test_linear_transform_validation_and_forward() -> None:
    width = 4
    transform = LinearTransform(weight=np.eye(width, dtype=np.float32), activation="relu")
    output = transform.forward(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))
    assert output.shape == (width,)
    assert np.all(output >= 0)


def test_linear_transform_rejects_bad_width() -> None:
    transform = LinearTransform(weight=np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="input width mismatch"):
        transform.forward(np.ones(2, dtype=np.float32))
