# LatentMesh

**Multi-agent KV-cache communication for LLMs.**

LatentMesh wires multiple LLM agents together in a [LangGraph](https://github.com/langchain-ai/langgraph) pipeline. When Agent A generates text, the full KV cache is stored in a `GlobalPrefixCache`. When Agent B runs next, LatentMesh finds the longest matching prefix and injects the cached KV state — so Agent B never re-encodes what Agent A already processed.

## Installation

```bash
pip install latentmesh
```

Or from source:

```bash
git clone https://github.com/shayhacker/LatentMesh.git
cd LatentMesh
pip install -e .
```

**Core requirements:** `torch`, `transformers`, `langgraph`, `langchain-core`, `pygtrie`

**Optional:** `pip install latentmesh[disk]` for persistent disk-backed caching, `pip install latentmesh[server]` for the FastAPI server.

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

## API Reference

| Module | Contents |
|---|---|
| `latentmesh.core` | `LatentLLM`, `AgentOutput`, `extract_kv` |
| `latentmesh.graph` | `LatentState`, `latent_reducer` |
| `latentmesh.primitives` | `AgentPrimitive`, `PlanPrimitive`, `ReasonPrimitive`, `ReviewPrimitive`, `VotingPrimitive` |
| `latentmesh.persistent_cache` | `GlobalPrefixCache`, `MemoryKVStore`, `DiskKVStore` |

### `LatentLLM(model_name, device="cuda", dtype="auto", global_cache=None, debug=False)`

Wraps any HuggingFace `AutoModelForCausalLM`.

- **`global_cache`**: A `GlobalPrefixCache` for automatic KV-cache reuse.
- **`debug`**: When `True`, logs cache hit/miss details and token counts.
- **`generate(messages, max_new_tokens, ...)`** → `AgentOutput`

### Primitives

All primitives are callable LangGraph nodes:

| Primitive | Default Trigger | Purpose |
|---|---|---|
| `PlanPrimitive(llm)` | `"Break the problem into clear steps..."` | Structural decomposition |
| `ReasonPrimitive(llm)` | `"Now reason through each step..."` | Core computation |
| `ReviewPrimitive(llm)` | `"Review the reasoning above..."` | Verification & refinement |
| `VotingPrimitive(name, candidates)` | — | Selects candidate with highest generation log-probability |

### Graph State

`LatentState` is a `TypedDict` with:

- `messages` — list of message dicts (accumulated via list concatenation)
- `latent` — `AgentOutput` with generated `text`, token counts, and diagnostics
- `tokens_so_far` — running total of generated tokens

## Examples

| Example | Description |
|---|---|
| [`sequential.py`](examples/sequential.py) | Plan → Reason → Review pipeline |
| [`complex.py`](examples/complex.py) | Multi-path voting with `VotingPrimitive` |
| [`hierarchical.py`](examples/hierarchical.py) | Supervisor routing based on generated text |

## Server

Start an OpenAI-compatible API server:

```bash
pip install latentmesh[server]
python -m latentmesh.server
```

## References

- [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)
- [Agent Primitives: Reusable Latent Building Blocks for MAS](https://arxiv.org/abs/2602.03695)

## License

MIT — see [LICENSE](LICENSE).