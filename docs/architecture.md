# Architecture

## Data Flow

```
User Prompt
    │
    ▼
┌──────────────────────────────────┐
│  Agent 1 (e.g., Planner)         │
│  1. Build messages (user + trigger)
│  2. Expand via chat template      │
│  3. GlobalPrefixCache.query()     │
│     → cache MISS (cold start)     │
│  4. Full encode + model.generate()│
│  5. Store KV cache in prefix cache│
│  6. Return AgentOutput with text  │
└────────────┬─────────────────────┘
             │  text appended to messages
             ▼
┌──────────────────────────────────┐
│  Agent 2 (e.g., Reasoner)        │
│  1. Build messages (user + A1 text + trigger)
│  2. Expand via chat template      │
│  3. GlobalPrefixCache.query()     │
│     → cache HIT (Agent 1's prefix)│
│  4. Encode only delta tokens      │
│  5. model.generate(past_key_values=cached)
│  6. Store updated KV cache        │
│  7. Return AgentOutput with text  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Agent 3 (e.g., Reviewer)        │
│  → cache HIT (Agent 2's prefix)  │
│  → encode only new delta         │
│  → Return final answer           │
└──────────────────────────────────┘
```

## GlobalPrefixCache

Backed by a `pygtrie.CharTrie` for O(L) longest-prefix lookup (where L is the query text length).

**Insert:** After each generation, the full text (prompt + output) is stored as a trie key, with a UUID pointing to the serialised KV bytes in the `KVStore`.

**Query:** Given a new prompt text, the trie finds the longest stored prefix. If the tokenisation of that prefix aligns with the start of the full token sequence, the cached KV state is loaded and only the delta tokens are encoded.

**Alignment check:** After finding a prefix match, `LatentLLM` re-tokenises the matched text and verifies `prefix_ids == full_ids[:len(prefix_ids)]`. This guards against tokeniser boundary mismatches where the same text tokenises differently in isolation vs. as part of a longer sequence.

## KV Cache Serialisation

KV caches are serialised via `torch.save()` into byte buffers:

```python
buffer = io.BytesIO()
torch.save(extract_kv(returned_kv), buffer)
cache.insert(full_text, buffer.getvalue())
```

`extract_kv()` normalises three HuggingFace KV formats (tuple, `DynamicCache`, layer-based) into a flat `[(k, v), ...]` list.

Deserialisation uses `DynamicCache.update()` to reconstruct the cache layer by layer with correct device/dtype placement.

## LangGraph Integration

`LatentState` is a `TypedDict` with LangGraph-compatible reducers:

- `messages: Annotated[List[dict], add]` — list concatenation
- `latent: Annotated[Optional[AgentOutput], latent_reducer]` — text concatenation, token summation
- `tokens_so_far: Annotated[int, add]` — integer addition

Each primitive node returns `{"latent": AgentOutput, "tokens_so_far": int}`, and LangGraph handles merging.

## Token Tracking

Every `AgentOutput` tracks three token counts:

| Field | Meaning |
|---|---|
| `input_tokens_uncached` | Tokens that had to be freshly encoded (cache miss portion) |
| `cached_tokens` | Tokens whose KV state was loaded from the cache |
| `output_tokens` | Tokens produced during generation |
