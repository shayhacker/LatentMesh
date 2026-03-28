"""
Sequential Pipeline Example — Plan → Reason → Review

Demonstrates a linear 3-agent pipeline where each agent's KV cache
is automatically reused by the next agent via GlobalPrefixCache.
"""

import torch
from langgraph.graph import StateGraph, START, END
from latentmesh import LatentLLM, LatentState
from latentmesh.primitives import PlanPrimitive, ReasonPrimitive, ReviewPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache

if __name__ == "__main__":
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    llm = LatentLLM(
        model_name="HuggingFaceTB/SmolLM-135M",
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_cache=cache,
        debug=True,
    )

    planner = PlanPrimitive(llm, max_new_tokens=24)
    reasoner = ReasonPrimitive(llm, max_new_tokens=24)
    reviewer = ReviewPrimitive(llm, max_new_tokens=24)

    builder = StateGraph(LatentState)

    builder.add_node("planner", planner)
    builder.add_node("reasoner", reasoner)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reasoner")
    builder.add_edge("reasoner", "reviewer")
    builder.add_edge("reviewer", END)

    graph = builder.compile()

    initial_text = "What is the capital of France and why is it important?"
    print(f"\n[User]: {initial_text}")

    initial_state = {
        "messages": [{"role": "user", "content": initial_text}],
        "tokens_so_far": 0,
    }

    print("\n[LatentMesh]: Running pipeline…")
    final_state = graph.invoke(initial_state)

    result = final_state["latent"].text
    total_tokens = final_state["tokens_so_far"]

    print(f"\n[Assistant]: {result}")
    print(f"[Tokens]: {total_tokens}\n")
