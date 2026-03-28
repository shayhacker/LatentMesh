"""
Agent primitives for LatentMesh.

Each primitive is a callable LangGraph node that wraps ``LatentLLM.generate()``
with a role-specific system prompt (trigger text).
"""

import logging
from typing import Any, Dict, List, Optional

from latentmesh.core import AgentOutput
from latentmesh.graph import LatentState

logger = logging.getLogger(__name__)


class AgentPrimitive:
    """
    Base LangGraph node that generates text with a role-specific trigger prompt.

    The trigger text is injected as an assistant message before generation,
    steering the model toward the desired reasoning behaviour.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        trigger_text: str = "",
        max_new_tokens: int = 128,
    ) -> None:
        """
        Args:
            name: Human-readable label for this agent (used in logging).
            llm: A ``LatentLLM`` instance.
            trigger_text: Prompt fragment prepended as an assistant message.
            max_new_tokens: Generation budget for this node.
        """
        self.name = name
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.trigger_text = trigger_text

    def __call__(self, state: LatentState) -> Dict[str, Any]:
        """Execute the primitive: build messages, generate, return state update."""
        past_latent = state.get("latent")
        messages = list(state.get("messages", []))

        # Append prior agent's generated text to the conversation history
        if past_latent and past_latent.text:
            messages.append({"role": "assistant", "content": past_latent.text})

        if self.trigger_text:
            messages.append({"role": "user", "content": self.trigger_text})

        result = self.llm.generate(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
        )

        return {
            "latent": result,
            "tokens_so_far": result.tokens,
        }


class PlanPrimitive(AgentPrimitive):
    """Decomposes the task into a structured step-by-step plan."""

    def __init__(
        self,
        llm: Any,
        trigger_text: str = "Break the problem into clear steps. Outline a plan without solving it:",
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            name="Planner",
            llm=llm,
            max_new_tokens=max_new_tokens,
            trigger_text=trigger_text,
        )


class ReasonPrimitive(AgentPrimitive):
    """Executes detailed reasoning following a plan or prompt."""

    def __init__(
        self,
        llm: Any,
        trigger_text: str = "Now reason through each step carefully and work toward the solution:",
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            name="Reasoner",
            llm=llm,
            max_new_tokens=max_new_tokens,
            trigger_text=trigger_text,
        )


class ReviewPrimitive(AgentPrimitive):
    """Verifies prior reasoning and produces a final answer."""

    def __init__(
        self,
        llm: Any,
        trigger_text: str = "Review the reasoning above for errors. Provide the verified final answer:",
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            name="Reviewer",
            llm=llm,
            max_new_tokens=max_new_tokens,
            trigger_text=trigger_text,
        )


class VotingPrimitive:
    """
    Runs multiple candidate agents on the same input and selects the best one
    based on the highest mean generation log-probability.
    """

    def __init__(self, name: str, candidates: List[AgentPrimitive]) -> None:
        if not candidates:
            raise ValueError("VotingPrimitive requires at least one candidate.")
        self.name = name
        self.candidates = candidates

    def __call__(self, state: LatentState) -> Dict[str, Any]:
        """Execute all candidates sequentially and return the highest-confidence result."""
        candidate_results: list[Dict[str, Any]] = []
        for cand in self.candidates:
            res = cand(state)
            candidate_results.append(res)

        best_idx = 0
        max_logprob = -float("inf")

        for i, res in enumerate(candidate_results):
            latent: Optional[AgentOutput] = res.get("latent")
            if latent is not None and latent.debug_text:
                logprob = -float("inf")
                for entry in latent.debug_text:
                    if entry.startswith("mean_logprob:"):
                        try:
                            logprob = float(entry.split(":")[1])
                        except ValueError:
                            pass
                if logprob > max_logprob:
                    max_logprob = logprob
                    best_idx = i

        selected = candidate_results[best_idx]
        logger.info(
            f"VotingPrimitive '{self.name}' selected candidate {best_idx} "
            f"(mean_logprob={max_logprob:.4f})"
        )
        return {
            "latent": selected["latent"],
            "tokens_so_far": selected["tokens_so_far"],
        }