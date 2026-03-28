"""
LangGraph state schema and reducer for LatentMesh.

Defines ``LatentState`` (the graph-level TypedDict) and the reducer that merges
``AgentOutput`` objects across sequential and parallel node boundaries.
"""

from operator import add
from typing import Annotated, Any, List, Optional, TypedDict

from latentmesh.core import AgentOutput


def latent_reducer(left: Optional[AgentOutput], right: Optional[AgentOutput]) -> AgentOutput:
    """
    LangGraph reducer for the ``latent`` field of ``LatentState``.

    Merging rules:
      - Text is concatenated with a newline separator.
      - Token counts are summed.
      - ``debug_text`` lists are concatenated.

    This supports both sequential handoff (agent after agent) and fan-in
    (parallel agents converging).
    """
    if right is None:
        return left if left is not None else AgentOutput()
    if left is None:
        return right

    if left.text is not None and right.text is not None:
        merged_text = left.text + "\n" + right.text
    elif right.text is not None:
        merged_text = right.text
    else:
        merged_text = left.text

    return AgentOutput(
        text=merged_text,
        debug_text=(left.debug_text or []) + (right.debug_text or []),
        tokens=left.tokens + right.tokens,
        input_tokens_uncached=left.input_tokens_uncached + right.input_tokens_uncached,
        cached_tokens=left.cached_tokens + right.cached_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
    )


class LatentState(TypedDict):
    """
    LangGraph state dictionary shared across all nodes in a LatentMesh graph.

    Fields:
        messages: Accumulated chat-format messages (reduced via list concatenation).
        latent: The most recent ``AgentOutput`` (reduced via ``latent_reducer``).
        tokens_so_far: Running total of generated tokens (reduced via addition).
    """
    messages: Annotated[List[dict], add]
    latent: Annotated[Optional[AgentOutput], latent_reducer]
    tokens_so_far: Annotated[int, add]
