# LatentMesh

## **Multi-agent Latent Space Communication**

LatentMesh wires multiple LLM agents together in a [LangGraph](https://github.com/langchain-ai/langgraph) pipeline. When Agent B is run, it never has to re-read Agent A's work.

## Installation

```bash
pip install latentmesh
```
> Not yet published!

Or from source:

```bash
git clone https://github.com/shayhacker/LatentMesh.git
cd LatentMesh
pip install -e .
```

**Optional:**
- For persistent disk-backed caching: `pip install latentmesh[disk]`
- For a FastAPI server: `pip install latentmesh[server]` 

## Quick Start

```python
from langgraph.graph import StateGraph, START, END
from latentmesh import LatentLLM, LatentState
from latentmesh.primitives import PlanPrimitive, ReasonPrimitive, ReviewPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache

# 1. Set up the cache
store = MemoryKVStore()
cache = GlobalPrefixCache(store)

# 2. Load any HuggingFace causal LM
llm = LatentLLM("Qwen/Qwen3-0.6B", device="cuda", global_cache=cache)

# 3. Create primitives (each is a LangGraph node)
planner  = PlanPrimitive(llm)
reasoner = ReasonPrimitive(llm)
reviewer = ReviewPrimitive(llm)

# 4. Build a LangGraph
builder = StateGraph(LatentState)
builder.add_node("planner", planner)
builder.add_node("reasoner", reasoner)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "planner")
builder.add_edge("planner", "reasoner")
builder.add_edge("reasoner", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()

# 5. Run
result = graph.invoke({
    "messages": [{"role": "user", "content": "What is the cosine of 45 degrees?"}],
    "tokens_so_far": 0,
})

print(result["latent"].text)
print(f"Total tokens: {result['tokens_so_far']}")
```

## Examples

| Example | Description |
|---|---|
| [`sequential.py`](examples/sequential.py) | Plan → Reason → Review pipeline |
| [`complex.py`](examples/complex.py) | Multi-path voting with `VotingPrimitive` |
| [`hierarchical.py`](examples/hierarchical.py) | Supervisor routing based on generated text |

## License

MIT [LICENSE](LICENSE).
