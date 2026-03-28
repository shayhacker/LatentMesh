# API Reference: `latentmesh.core`

## `AgentOutput`

```python
@dataclass
class AgentOutput:
    text: Optional[str] = None
    debug_text: Optional[List[str]] = None
    tokens: int = 0
    input_tokens_uncached: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
```

Result container returned by `LatentLLM.generate()`.

| Field | Description |
|-------|-------------|
| `text` | Generated text |
| `debug_text` | Diagnostic entries (e.g. `"mean_logprob:-0.5"`) |
| `tokens` | Number of generated tokens |
| `input_tokens_uncached` | Tokens freshly encoded (cache miss) |
| `cached_tokens` | Tokens loaded from GlobalPrefixCache |
| `output_tokens` | Tokens produced during generation |

---

## `LatentLLM`

```python
LatentLLM(model_name, device="cuda", dtype="auto", global_cache=None, debug=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | — | HuggingFace model ID or local path |
| `device` | `str` | `"cuda"` | `"cuda"` or `"cpu"` |
| `dtype` | `str` | `"auto"` | Torch dtype name (e.g. `"float16"`) |
| `global_cache` | `GlobalPrefixCache` | `None` | Cache instance for KV reuse |
| `debug` | `bool` | `False` | Log cache hits/misses and token counts |

### `generate()`

```python
def generate(messages, max_new_tokens=128, temperature=0.6, top_k=30000, top_p=0.95, output_scores=False) -> AgentOutput
```

| Parameter | Description |
|-----------|-------------|
| `messages` | Chat-format message list |
| `max_new_tokens` | Maximum tokens to generate |
| `output_scores` | Compute mean logprob in `debug_text` |

---

## `extract_kv(kv_cache) -> list`

Normalises HuggingFace KV cache formats into `[(key_tensor, value_tensor), ...]`.
