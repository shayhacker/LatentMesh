# LatentMesh Overview

LatentMesh is a LangGraph-based multi-agent system for LLMs. Agents share a `GlobalPrefixCache` so that when one agent generates text, the full KV cache is stored and the next agent can skip re-encoding the overlapping context.

## How It Works

1. **Agent A generates text.** The full KV cache (covering the prompt + generated tokens) is serialised and stored in `GlobalPrefixCache`, keyed by the complete expanded text.

2. **Agent B runs next.** Its prompt includes Agent A's output as part of the message history. `LatentLLM.generate()` expands the full text, then calls `GlobalPrefixCache.query()` to find the longest stored text prefix that matches the start of the new prompt.

3. **Delta encoding.** Only the new tokens (not covered by the cache hit) are actually encoded by the transformer. The cached KV state is loaded and used as `past_key_values`, skipping re-computation.

4. **Repeat.** Each agent's output grows the cache, so subsequent agents benefit from progressively larger cache hits.

## Key Components

| Module | Class | Purpose |
|---|---|---|
| `latentmesh.core` | `LatentLLM` | HuggingFace model wrapper with automatic KV-cache management |
| `latentmesh.core` | `AgentOutput` | Generation result: text, token counts, diagnostics |
| `latentmesh.graph` | `LatentState` | LangGraph state schema (messages, latent, tokens_so_far) |
| `latentmesh.primitives` | `AgentPrimitive` | Base LangGraph node with trigger-text steering |
| `latentmesh.persistent_cache` | `GlobalPrefixCache` | Trie-indexed text → KV-cache mapping |
| `latentmesh.persistent_cache` | `MemoryKVStore` | In-memory storage backend (testing) |
| `latentmesh.persistent_cache` | `DiskKVStore` | Disk-backed storage via `diskcache` (production) |

## Design Principles

- **No silent fallbacks.** Errors are raised immediately. No try/except swallowing.
- **Model-agnostic.** Works with any `AutoModelForCausalLM` from HuggingFace.
- **LangGraph-native.** All agents are standard LangGraph nodes supporting conditional edges and fan-out.
