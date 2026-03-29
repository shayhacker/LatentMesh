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
        messages = list(state.get("messages", []))

        # We will append to THIS agent's local messages array
        new_messages = []

        # If a graph node previously updated latent but didn't return messages,
        # we append it just in case to maintain backward compatibility.
        past_latent = state.get("latent")
        if past_latent and past_latent.text:
            # Only append if the last message in state isn't already this latent text
            if not (messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == past_latent.text):
                new_messages.append({"role": "assistant", "content": past_latent.text})

        if self.trigger_text:
            new_messages.append({"role": "user", "content": self.trigger_text})

        full_messages = messages + new_messages

        # Merge contiguous roles to prevent chat template hallucination (e.g. back-to-back user prompts)
        merged_messages = []
        for msg in full_messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                merged_messages[-1] = {
                    "role": msg["role"],
                    "content": merged_messages[-1]["content"] + "\n\n" + msg["content"]
                }
            else:
                merged_messages.append(msg)

        result = self.llm.generate(
            messages=merged_messages,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
        )

        new_messages.append({"role": "assistant", "content": result.text})

        return {
            "messages": new_messages,
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
            "messages": selected.get("messages", []),
            "latent": selected["latent"],
            "tokens_so_far": selected["tokens_so_far"],
        }


class RouterPrimitive(AgentPrimitive):
    """
    Evaluates the input and selects the best route from a list of candidates.
    Outputs the decision without appending its reasoning to the message history, 
    ensuring downstream specialists don't inherit the routing context.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        routes: List[tuple[str, str]],
        max_new_tokens: int = 1024,
    ) -> None:
        self.routes = routes
        self._route_ids = [r[1] for r in routes]
        self._default_route = self._route_ids[0] if self._route_ids else "default"
        
        # Build the dynamic routing prompt
        prompt_lines = [
            "Classify the request into the most appropriate category and route it to the correct specialist.",
            "Choose exactly one of the following specialists:"
        ]
        for desc, route_id in routes:
            prompt_lines.append(f"- '{route_id}': {desc}")
            
        prompt_lines.append(
            "\nCRITICAL INSTRUCTION: You must analyze the input and then provide your FINAL decision by wrapping EXACTLY the Route ID in [[[]]]."
            "\n"
            "\nExample Response Format:"
            "\n..."
            "\n[[[route_id]]]"
            "\n"
            "\nYou must not include any other text. Your answer must include this EXACT format with NO deviations. DO NOT SOLVE THE PROBLEM. YOU MUST ROUTE THIS PROBLEM TO THE SPECIALIST."
        )
        
        super().__init__(
            name=name,
            llm=llm,
            max_new_tokens=max_new_tokens,
            trigger_text="\n".join(prompt_lines),
        )

    def __call__(self, state: LatentState) -> Dict[str, Any]:
        """Execute the primitive, but DO NOT append its output to messages."""
        res = super().__call__(state)
        # Clear messages so the router's thoughts/trigger don't pollute downstream context.
        # Downstream nodes will read the original user prompt exactly as it was.
        res["messages"] = []
        
        import re
        text = res["latent"].text
        # Find all occurrences of [[[...]]]
        matches = re.findall(r"\[\[\[(.*?)\]\]\]", text, flags=re.IGNORECASE | re.DOTALL)
        
        decision = self._default_route
        if matches:
            parsed = matches[-1].strip()
            if parsed in self._route_ids:
                decision = parsed
            
        # Hide the raw text so downstream agents do not auto-append it as an assistant message
        res["latent"].text = ""
        
        # Stash the parsed decision in debug_text for the routing condition to safely read
        res["latent"].debug_text = (res["latent"].debug_text or []) + [f"ROUTE:{decision}"]
        
        return res

    def route_condition(self, state: LatentState) -> str:
        """LangGraph conditional edge function to extract the parsed route."""
        latent = state.get("latent")
        if latent is not None and latent.debug_text:
            route_decision = next((t for t in reversed(latent.debug_text) if t.startswith("ROUTE:")), None)
            if route_decision:
                decision = route_decision.replace("ROUTE:", "").strip()
                if decision in self._route_ids:
                    return decision
        return self._default_route