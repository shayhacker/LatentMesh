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
        "HuggingFaceTB/SmolLM-135M",
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_cache=cache,
        debug=True,
    )

    r1 = ReasonPrimitive(llm, trigger_text="Analyze this problem step by step:", max_new_tokens=24)
    r2 = ReasonPrimitive(llm, trigger_text="Think about this laterally and creatively:", max_new_tokens=24)
    r3 = ReasonPrimitive(llm, trigger_text="Take a conservative, careful approach:", max_new_tokens=24)

    consensus_node = VotingPrimitive("Consensus", candidates=[r1, r2, r3])

    builder = StateGraph(LatentState)
    builder.add_node("consensus", consensus_node)
    builder.add_edge(START, "consensus")
    builder.add_edge("consensus", END)

    graph = builder.compile()

    print("Running 3 reasoning branches and selecting the best…")
    final_state = graph.invoke({
        "messages": [{"role": "user", "content": "Solve the trolley problem."}],
        "tokens_so_far": 0,
    })

    winner = final_state["latent"]
    print(f"\n[Selected answer]: {winner.text}")
    print(f"[Tokens]: {final_state['tokens_so_far']}")