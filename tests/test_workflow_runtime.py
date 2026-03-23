from __future__ import annotations

import numpy as np
import pytest

from latentmesh import LatentLLM, LatentGraph


def build_workflow():
    llm = LatentLLM("mock://dev", backend="mock")
    return (
        LatentGraph(llm=llm, retrieval_k=2)
        .add_stage("planner", transform=lambda x: np.tanh(x))
        .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
        .add_stage("writer")
        .connect("planner", "critic")
        .connect("critic", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )


def test_invoke_returns_string() -> None:
    workflow = build_workflow()
    out = workflow.invoke("How do I rollback a failed deploy?")
    assert isinstance(out, str)
    assert out


def test_run_returns_trace() -> None:
    workflow = build_workflow()
    answer, trace = workflow.run("How do I rollback a failed deploy?")
    assert isinstance(answer, str)
    assert [step.stage for step in trace.steps] == ["planner", "critic", "writer"]
    assert trace.output_latent.ndim == 1


def test_batch_returns_outputs() -> None:
    workflow = build_workflow()
    outputs = workflow.batch(["Q1", "Q2", "Q3"])
    assert len(outputs) == 3
    assert all(isinstance(text, str) and text for text in outputs)


def test_retrieval_example_is_used() -> None:
    workflow = build_workflow()
    workflow.add_example("How do I rollback a failed deploy?", "Rollback to the previous healthy release.")
    _, trace = workflow.run("How do I rollback a failed deploy?")
    assert trace.retrieved_examples
    assert trace.retrieved_examples[0][0] == "How do I rollback a failed deploy?"


def test_empty_question_is_rejected() -> None:
    workflow = build_workflow()
    with pytest.raises(ValueError, match="non-empty"):
        workflow.invoke("   ")


def test_incompatible_merge_width_rejected() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    workflow = (
        LatentGraph(llm=llm)
        .add_stage("a")
        .add_stage("b", transform=lambda x: x[:128])
        .add_stage("writer")
        .connect("a", "writer")
        .connect("b", "writer")
        .compile(entry_stage="a", exit_stage="writer")
    )

    with pytest.raises(ValueError, match="incompatible latent widths"):
        workflow.invoke("Q")


def test_stage_output_rank_validation() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    workflow = (
        LatentGraph(llm=llm)
        .add_stage("planner", transform=lambda x: x.reshape(1, -1))
        .add_stage("writer")
        .connect("planner", "writer")
        .compile(entry_stage="planner", exit_stage="writer")
    )

    with pytest.raises(ValueError, match="rank-1"):
        workflow.invoke("Q")
