"""
LatentMesh — LangGraph-based multi-agent KV-cache communication protocol.

Agents share a ``GlobalPrefixCache`` so that downstream agents skip
re-encoding text that upstream agents already processed.
"""

from latentmesh.core import (
    AgentOutput,
    LatentLLM,
    extract_kv,
)

from latentmesh.graph import (
    LatentState,
    latent_reducer,
)

from latentmesh.primitives import (
    AgentPrimitive,
    PlanPrimitive,
    ReasonPrimitive,
    ReviewPrimitive,
    VotingPrimitive,
)

__version__ = "0.5.1"

__all__ = [
    "AgentOutput",
    "LatentLLM",
    "extract_kv",
    "LatentState",
    "latent_reducer",
    "AgentPrimitive",
    "PlanPrimitive",
    "ReasonPrimitive",
    "ReviewPrimitive",
    "VotingPrimitive",
]
