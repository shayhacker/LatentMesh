# LangGraph + RAG Integration

LatentMesh can be embedded into an existing LangGraph app instead of replacing it.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transformers,langgraph]"
```

## Run examples

1. Embed LatentGraph as a node in a regular LangGraph flow:

```bash
python3 examples/integrations/11_langgraph_as_node.py --model meta-llama/Llama-3.1-8B-Instruct
```

2. Use retrieval node + latent answer node (RAG style):

```bash
python3 examples/integrations/12_langgraph_rag.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --k 2
```

These examples keep all inter-stage communication latent while still using standard LangGraph graph orchestration.

