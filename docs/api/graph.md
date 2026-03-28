# API Reference: `latentmesh.graph`

## `LatentState`

```python
class LatentState(TypedDict):
    messages: Annotated[List[dict], add]
    latent: Annotated[Optional[AgentOutput], latent_reducer]
    tokens_so_far: Annotated[int, add]
```

LangGraph state schema. Fields are merged automatically by LangGraph using their annotated reducers.

| Field | Reducer | Description |
|-------|---------|-------------|
| `messages` | `add` (list concat) | Accumulated chat messages |
| `latent` | `latent_reducer` | Most recent `AgentOutput` |
| `tokens_so_far` | `add` (int sum) | Running total of generated tokens |

---

## `latent_reducer(left, right) -> AgentOutput`

Merges two `AgentOutput` instances:

- Text is concatenated with `"\n"`.
- Token counts are summed (total, cached, uncached, output).
- `debug_text` lists are concatenated.
- If either side is `None`, the other is returned.
