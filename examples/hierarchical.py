"""
Hierarchical Supervisor/Worker Example

A Supervisor agent generates initial reasoning, then a conditional edge
routes to a specialist based on the content of the generated text.
"""

import torch
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from latentmesh import LatentState, LatentLLM
from latentmesh.primitives import AgentPrimitive, ReasonPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache


class SupervisorNode:
    """Routes the task to a specialist by generating initial context."""

    def __init__(self, llm: LatentLLM) -> None:
        self.primitive = AgentPrimitive(
            "Supervisor", llm,
            trigger_text="Classify this problem and route to the right specialist:",
            max_new_tokens=16,
        )

    def __call__(self, state: LatentState) -> Dict[str, Any]:
        return self.primitive(state)


def routing_condition(state: LatentState) -> str:
    """Route based on keywords in the supervisor's generated text."""
    latent = state.get("latent")
    if latent is None or latent.text is None:
        return "creative_specialist"

    text_lower = latent.text.lower()
    math_keywords = ["math", "calcul", "number", "equation", "formula", "solve"]
    if any(kw in text_lower for kw in math_keywords):
        return "math_specialist"
    return "creative_specialist"


if __name__ == "__main__":
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    llm = LatentLLM(
        "HuggingFaceTB/SmolLM-135M",
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_cache=cache,
        debug=True,
    )

    supervisor = SupervisorNode(llm)
    math_specialist = ReasonPrimitive(
        llm,
        trigger_text="Solve this with rigorous mathematical reasoning:",
        max_new_tokens=32,
    )
    creative_specialist = ReasonPrimitive(
        llm,
        trigger_text="Approach this with creative brainstorming:",
        max_new_tokens=32,
    )

    builder = StateGraph(LatentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("math_specialist", math_specialist)
    builder.add_node("creative_specialist", creative_specialist)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", routing_condition)
    builder.add_edge("math_specialist", END)
    builder.add_edge("creative_specialist", END)

    graph = builder.compile()

    state = graph.invoke({
        "messages": [{"role": "user", "content": "Calculate the trajectory of a rocket."}],
        "tokens_so_far": 0,
    })

    print(f"\n[Final answer]: {state['latent'].text}")
    print(f"[Tokens]: {state['tokens_so_far']}")
