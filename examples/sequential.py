"""
Sequential Pipeline Example — Plan → Reason → Review

Demonstrates a linear 3-agent pipeline where each agent's KV cache
is automatically reused by the next agent via GlobalPrefixCache.
"""

import torch
from langgraph.graph import StateGraph, START, END
from latentmesh import LatentLLM, LatentState
from latentmesh.primitives import PlanPrimitive, ReasonPrimitive, ReviewPrimitive, AgentPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache

if __name__ == "__main__":
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    llm = LatentLLM(
        model_name="Qwen/Qwen3-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_cache=cache,
        debug=True,
    )

    planner = PlanPrimitive(llm, max_new_tokens=1024)
    reasoner = ReasonPrimitive(llm, max_new_tokens=1024)
    reviewer = ReviewPrimitive(llm, max_new_tokens=1024)
    generator = AgentPrimitive("generator", llm, max_new_tokens=1024)

    builder = StateGraph(LatentState)

    builder.add_node("planner", planner)
    builder.add_node("reasoner", reasoner)
    builder.add_node("reviewer", reviewer)
    builder.add_node("generator", generator)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reasoner")
    builder.add_edge("reasoner", "reviewer")
    builder.add_edge("reviewer", "generator")
    builder.add_edge("generator", END)

    graph = builder.compile()

    initial_text = "What is the capital of France and why is it important?"
    print(f"\n[User]: {initial_text}")

    initial_state = {
        "messages": [{"role": "user", "content": initial_text}],
        "tokens_so_far": 0,
    }

    print("\n[LatentMesh]: Running pipeline…")
    
    total_tokens = 0
    for event in graph.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n--- [Node: {node_name.upper()}] ---")
            
            latent = state_update.get("latent")
            if latent and latent.text:
                print(latent.text.strip())
                print(f"\n[Metrics]: Input Tokens (Cached: {latent.cached_tokens}, Uncached: {latent.input_tokens_uncached}) | Output Tokens: {latent.output_tokens}")
                
            if "tokens_so_far" in state_update:
                total_tokens += state_update["tokens_so_far"]

    print(f"\n=========================================")
    print(f"[Total Pipeline Tokens]: {total_tokens}\n")
