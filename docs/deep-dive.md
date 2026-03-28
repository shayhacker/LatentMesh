# Deep Dive: Implementation Details

## `LatentLLM.generate()` Step-by-Step

### 1. Prompt Expansion

```python
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
full_input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
```

The entire message history is rendered through the chat template (ChatML if no template is set), then tokenised.

### 2. Prefix Cache Lookup

```python
best_match_text, cached_bytes = global_cache.query(full_text)
```

The `CharTrie.longest_prefix()` finds the longest stored text that `full_text` starts with. This is O(L) where L is the length of `full_text`.

### 3. Tokenisation Alignment

```python
prefix_ids = tokenizer(best_match_text, ...).input_ids
if torch.equal(prefix_ids[0], full_input_ids[0, :matched_tokens]):
    past_key_values = _prepare_past_kv(cached_bytes)
```

Re-tokenising the matched prefix text and comparing against the full sequence prevents edge cases where tokeniser boundaries shift.

### 4. Delta Encoding

```python
input_ids = full_input_ids[:, matched_tokens:]
```

Only tokens not covered by the cache are passed to the model. For a 500-token prompt where 400 tokens are cached, only 100 tokens need encoding.

### 5. Position Management

```python
past_length = past_key_values.get_seq_length()
attention_mask = torch.ones((1, past_length + current_length), device=device)
cache_position = torch.arange(past_length, past_length + current_length, device=device)
```

The attention mask spans the full sequence (cached + new). `cache_position` tells the model the absolute position of each new token so positional encodings (RoPE, etc.) are applied correctly.

### 6. Generation

Standard `model.generate()` with the pre-filled KV cache. `output_scores=True` is always on to support `VotingPrimitive`'s logprob-based selection.

### 7. Cache Storage

```python
new_full_text = full_text + generated_text
buffer = io.BytesIO()
torch.save(extract_kv(returned_kv), buffer)
global_cache.insert(new_full_text, buffer.getvalue())
```

The complete text (prompt + generation) is stored so the next agent in the pipeline can hit this cache.

---

## VotingPrimitive

Runs all candidates **sequentially** on the same input state. Each candidate's `AgentOutput.debug_text` contains a `"mean_logprob:<value>"` entry computed from `torch.softmax(scores).log().mean()`. The candidate with the highest mean logprob is selected.

---

## Storage Backends

| Backend | Backing Store | Use Case |
|---------|---------------|----------|
| `MemoryKVStore` | Python `dict` | Tests, single-process |
| `DiskKVStore` | `diskcache.Cache` (SQLite + files) | Production, persistent |

Both implement the same `KVStore` protocol: `store(key, bytes)`, `load(key) -> bytes`, `delete(key)`.

`MemoryKVStore.load()` raises `KeyError` on missing keys. `DiskKVStore.load()` raises `KeyError` on missing keys. No silent fallbacks.
