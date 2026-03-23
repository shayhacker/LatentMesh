# LatentMesh

LangGraph-first latent agent framework.

You build a standard LangGraph workflow, but every internal stage exchanges only latent vectors. Text is only used at the ingress (user input) and egress (final answer).

## Core API (OOP)

- `LatentLLM("hf/model")`
- `LatentGraph(...)`
- `LatentStage` + `LatentTransform`
- `LatentWorkflow`
- `LatentServe`

## Install

```bash
python3 -m pip install -e .
```

Optional extras:

```bash
python3 -m pip install -e '.[transformers]'
python3 -m pip install -e '.[ollama]'
python3 -m pip install -e '.[langgraph]'
python3 -m pip install -e '.[serve]'
```

## Build a Latent LangGraph

```python
import numpy as np
from latentmesh import LatentLLM, LatentGraph

llm = LatentLLM("mock://dev", backend="mock")
# production example:
# llm = LatentLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

workflow = (
    LatentGraph(llm=llm, name="incident-response", retrieval_k=2)
    .add_stage("planner", transform=lambda x: np.tanh(x))
    .add_stage("critic", transform=lambda x: np.maximum(0.0, x))
    .add_stage("writer")
    .connect("planner", "critic")
    .connect("critic", "writer")
    .add_example("How to rollback deploy?", "Rollback to last healthy build and run smoke tests.")
    .compile(entry_stage="planner", exit_stage="writer")
)

print(workflow.invoke("Production deploy failed after migration timeout."))
```

## Serve with LangServe or OpenAI-compatible API

```python
from latentmesh import LatentServe

gateway = LatentServe(workflow, model_name="latent-langgraph")

# One app with both:
# - LangServe routes under /latent
# - OpenAI endpoints under /v1/*
gateway.serve(mode="unified", host="0.0.0.0", port=8000, langserve_path="/latent")
```

## Existing ecosystem integration

- **LangGraph runtime**: used as the workflow execution backend when installed.
- **LangServe**: exposes `invoke`/`batch` style API for direct integration.
- **OpenAI-compatible endpoint**: `/v1/chat/completions` and `/v1/models` for tooling compatibility.

## Examples

- `/Users/shayhacker/Desktop/personal/DeepSync/examples/README.md`
- One-GPU HF: `/Users/shayhacker/Desktop/personal/DeepSync/examples/hf`
- Ollama local: `/Users/shayhacker/Desktop/personal/DeepSync/examples/ollama`
- Serving: `/Users/shayhacker/Desktop/personal/DeepSync/examples/serving`
- Integrations (LangGraph, LangChain, OpenAI SDK, Claude MCP): `/Users/shayhacker/Desktop/personal/DeepSync/examples/integrations`
- Ops smoke scripts: `/Users/shayhacker/Desktop/personal/DeepSync/examples/ops`

## Docs Website (GitHub Pages)

- `docs/index.html` (Home)
- `docs/docs.html` (Docs)
- `docs/benchmarks.html` (Benchmarks)

## Tests

```bash
python3 -m pytest
```

All tests are model-free and do not load LLMs.
