# One-GPU Local Models

Use these examples with a single GPU and Hugging Face checkpoints.

## 1) Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transformers,langgraph,serve]"
```

## 2) Run direct workflows

```bash
python3 examples/hf/01_deepseek_r1_single_agent.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
python3 examples/hf/02_llama31_single_agent.py --model meta-llama/Llama-3.1-8B-Instruct
python3 examples/hf/03_deepseek_r1_multi_stage.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 3) Serve OpenAI-compatible endpoint

```bash
python3 examples/serving/08_serve_openai.py --model meta-llama/Llama-3.1-8B-Instruct --port 8000
python3 examples/serving/10_client_smoke_openai.py --url http://127.0.0.1:8000/v1/chat/completions
```

## 4) Run ops smoke

```bash
python3 examples/ops/15_health_probe.py --base-url http://127.0.0.1:8000
python3 examples/ops/16_concurrent_smoke.py --url http://127.0.0.1:8000/v1/chat/completions
```

