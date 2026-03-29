"""
Hierarchical Supervisor/Worker Example

A Supervisor agent generates initial reasoning, then a conditional edge
routes to a specialist based on the content of the generated text.
"""

import torch
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from latentmesh import LatentState, LatentLLM
from latentmesh.primitives import AgentPrimitive, ReasonPrimitive, RouterPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache


if __name__ == "__main__":
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    llm = LatentLLM(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_cache=cache,
        debug=True,
    )

    supervisor = RouterPrimitive(
        name="Router",
        llm=llm,
        routes=[
            ("complex mathematical problems or calculations", "math_specialist"),
            ("creative writing, generative tasks, or open-ended brainstorming", "creative_specialist")
        ],
        max_new_tokens=1024,
    )

    math_specialist = ReasonPrimitive(
        llm,
        trigger_text="Solve this with rigorous mathematical reasoning:",
        max_new_tokens=1024,
    )
    
    creative_specialist = ReasonPrimitive(
        llm,
        trigger_text="Approach this with creative brainstorming:",
        max_new_tokens=1024,
    )

    builder = StateGraph(LatentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("math_specialist", math_specialist)
    builder.add_node("creative_specialist", creative_specialist)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", supervisor.route_condition)
    builder.add_edge("math_specialist", END)
    builder.add_edge("creative_specialist", END)

    graph = builder.compile()

    initial_text = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"
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
            
            if latent:
                print(
                    f"\n[Metrics]: Cached Input: {latent.cached_tokens}, Uncached Input: {latent.input_tokens_uncached}, Output: {latent.output_tokens}"
                )
                
            if "tokens_so_far" in state_update:
                total_tokens += state_update["tokens_so_far"]

    print("=========================================")
    print(f"[Total Pipeline Tokens]: {total_tokens}\n")
