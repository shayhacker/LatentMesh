"""
Multi-Path Voting Example

Three reasoning agents with different trigger prompts compete on the same
input.  ``VotingPrimitive`` selects the candidate with the highest mean
generation log-probability (proxy for confidence).
"""

import torch
from langgraph.graph import END, START, StateGraph
from latentmesh import LatentState, LatentLLM
from latentmesh.primitives import ReasonPrimitive, VotingPrimitive
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

    r1 = ReasonPrimitive(llm, trigger_text="Analyze this problem step by step:", max_new_tokens=1024)
    r2 = ReasonPrimitive(llm, trigger_text="Think about this laterally and creatively:", max_new_tokens=1024)
    r3 = ReasonPrimitive(llm, trigger_text="Take a conservative, careful approach:", max_new_tokens=1024)

    consensus_node = VotingPrimitive("Consensus", candidates=[r1, r2, r3])

    builder = StateGraph(LatentState)
    builder.add_node("consensus", consensus_node)
    builder.add_edge(START, "consensus")
    builder.add_edge("consensus", END)

    graph = builder.compile()

    initial_text = "A user asks for information that is factually correct but could be used to harm others. Do you provide it?"
    initial_state = {
        "messages": [{"role": "user", "content": initial_text}],
        "tokens_so_far": 0,
    }

    print("\n[LatentMesh]: Running 3 reasoning branches…")
    
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