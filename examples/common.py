"""Shared helpers for top-level LatentMesh examples."""

from __future__ import annotations

import numpy as np

from latentmesh import LatentGraph, LatentLLM, LatentRunTrace


def build_default_flow(
    llm: LatentLLM,
    *,
    name: str = "default-flow",
    retrieval_k: int = 3,
):
    """Build a practical 3-stage latent workflow for production examples."""

    flow = (
        LatentGraph(llm=llm, name=name, retrieval_k=retrieval_k)
        .add_stage("plan", transform=lambda x: np.tanh(x))
        .add_stage("crit", transform=lambda x: np.maximum(0.0, x))
        .add_stage("write")
        .connect("plan", "crit")
        .connect("crit", "write")
        .compile(entry_stage="plan", exit_stage="write")
    )
    return flow


def seed_sre_memory(flow) -> None:
    """Seed minimal but realistic runbook examples for retrieval guidance."""

    flow.add_example(
        "How do I rollback a failed deploy?",
        "Rollback to last healthy release, validate core endpoints, and hold traffic until error budget stabilizes.",
    )
    flow.add_example(
        "How should I respond to migration timeouts?",
        "Stop writes, verify lock contention, roll back schema change, then re-run with phased rollout.",
    )
    flow.add_example(
        "What should happen after emergency rollback?",
        "Create incident timeline, add prevention tasks, and gate next deploy on postmortem action items.",
    )


def print_trace(trace: LatentRunTrace) -> None:
    """Print compact trace info for stage-level debugging."""

    print("stages:", [s.stage for s in trace.steps])
    print("norms:", [(round(s.input_norm, 4), round(s.output_norm, 4)) for s in trace.steps])
    print("retrieved:", trace.retrieved_examples)
