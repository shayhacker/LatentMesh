# Claude Integration (Local Models via MCP Tool)

This path keeps local models in LatentMesh and exposes them to Claude through a local MCP tool server.

## 1) Start LatentMesh server

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[transformers,serve]"
python3 examples/serving/08_serve_openai.py --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

## 2) Start MCP bridge

```bash
python3 -m pip install mcp
python3 examples/integrations/15_claude_mcp_server.py
```

The MCP tool is `latent_chat(prompt, base_url, model, max_tokens, temperature)`.

## 3) Register in Claude Desktop config

Add this to Claude Desktop MCP config (adjust absolute paths):

```json
{
  "mcpServers": {
    "latentmesh": {
      "command": "python3",
      "args": [
        "/Users/shayhacker/Desktop/personal/DeepSync/examples/integrations/15_claude_mcp_server.py"
      ]
    }
  }
}
```

Restart Claude Desktop, then call the `latent_chat` tool from Claude.

