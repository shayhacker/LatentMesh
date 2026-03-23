# LatentMesh Examples

These are production-style, runnable examples for one-GPU local models and real integrations.

Run from repo root: `/Users/shayhacker/Desktop/personal/DeepSync`.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transformers,ollama,langgraph,serve]"
```

Optional client integrations:

```bash
python3 -m pip install langchain-openai openai
python3 -m pip install mcp
```

## One-GPU HF Workflows

```bash
python3 examples/hf/01_deepseek_r1_single_agent.py
python3 examples/hf/02_llama31_single_agent.py
python3 examples/hf/03_deepseek_r1_multi_stage.py
python3 examples/hf/04_memory_seeded_support_agent.py
python3 examples/hf/05_batch_eval.py
```

## Ollama Local Workflows

```bash
python3 examples/ollama/06_ollama_llama_local.py
python3 examples/ollama/07_ollama_multi_stage_retrieval.py
```

## Serving

```bash
python3 examples/serving/08_serve_openai.py
python3 examples/serving/09_serve_unified.py
python3 examples/serving/10_client_smoke_openai.py
```

## Integrations

```bash
python3 examples/integrations/11_langgraph_as_node.py
python3 examples/integrations/12_langgraph_rag.py
python3 examples/integrations/13_langchain_openai_client.py
python3 examples/integrations/14_openai_sdk_client.py
python3 examples/integrations/15_claude_mcp_server.py
```

## Ops

```bash
python3 examples/ops/15_health_probe.py
python3 examples/ops/16_concurrent_smoke.py
python3 examples/ops/17_jsonl_batch_client.py --in prompts.jsonl --out outputs.jsonl
```

## Guides

- `examples/guides/one_gpu_hf.md`
- `examples/guides/langgraph_rag.md`
- `examples/guides/claude_cli_local.md`

