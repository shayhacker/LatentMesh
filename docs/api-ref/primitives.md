# API Reference: `latentmesh.primitives`

## `AgentPrimitive`

```python
AgentPrimitive(name, llm, trigger_text="", max_new_tokens=128)
```

Base LangGraph node. Builds messages from the current state, appends trigger text, calls `llm.generate()`, and returns `{"latent": AgentOutput, "tokens_so_far": int}`.

| Parameter | Description |
|-----------|-------------|
| `name` | Human-readable label (for logging) |
| `llm` | `LatentLLM` instance |
| `trigger_text` | Prompt fragment added before generation |
| `max_new_tokens` | Generation budget |

---

## `PlanPrimitive`

```python
PlanPrimitive(llm, trigger_text="Break the problem into clear steps...", max_new_tokens=128)
```

Structural decomposition agent. Steers the model toward planning without solving.

---

## `ReasonPrimitive`

```python
ReasonPrimitive(llm, trigger_text="Now reason through each step carefully...", max_new_tokens=128)
```

Detailed reasoning agent. Works through the plan step by step.

---

## `ReviewPrimitive`

```python
ReviewPrimitive(llm, trigger_text="Review the reasoning above for errors...", max_new_tokens=128)
```

Verification agent. Checks prior reasoning and produces a final answer.

---

## `VotingPrimitive`

```python
VotingPrimitive(name, candidates)
```

Runs all candidate `AgentPrimitive` instances on the same input and selects the one with the highest mean generation log-probability.

| Parameter | Description |
|-----------|-------------|
| `name` | Label for logging |
| `candidates` | List of `AgentPrimitive` instances to compete |
