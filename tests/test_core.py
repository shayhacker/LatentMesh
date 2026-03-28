import torch
import pytest
from latentmesh.core import LatentLLM, AgentOutput, extract_kv

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_latent_llm_initialization_dtypes(dtype):
    """LatentLLM loads with the correct torch dtype."""
    llm = LatentLLM(model_name=MODEL_NAME, device="cpu", dtype=dtype)
    assert llm.model_dtype == getattr(torch, dtype)


def test_extract_kv_tuple_format():
    """extract_kv handles legacy tuple-of-tuples format."""
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    legacy = ((k, v), (k, v))
    extracted = extract_kv(legacy)
    assert len(extracted) == 2
    assert torch.equal(extracted[0][0], k)


def test_extract_kv_dynamic_cache_format():
    """extract_kv handles DynamicCache-like objects with key_cache/value_cache."""
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)

    class MockCache:
        def __init__(self):
            self.key_cache = [k, k]
            self.value_cache = [v, v]

    extracted = extract_kv(MockCache())
    assert len(extracted) == 2
    assert torch.equal(extracted[0][1], v)


def test_extract_kv_none():
    """extract_kv returns [] for None input."""
    assert extract_kv(None) == []


def test_agent_output_creation():
    """AgentOutput initialises with correct defaults."""
    out = AgentOutput(tokens=10)
    assert out.tokens == 10
    assert out.debug_text == []
    assert out.text is None
    assert "AgentOutput" in repr(out)


@pytest.mark.parametrize("max_tokens", [1, 5])
def test_generate_basic(max_tokens):
    """Basic generation returns an AgentOutput with text and token count."""
    llm = LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32")
    messages = [{"role": "user", "content": "Test prompt"}]
    result = llm.generate(messages=messages, max_new_tokens=max_tokens)

    assert isinstance(result, AgentOutput)
    assert isinstance(result.text, str)
    assert result.tokens > 0
    assert result.output_tokens > 0
    assert result.input_tokens_uncached > 0


def test_generate_with_scores():
    """Generation with output_scores=True populates debug_text with mean_logprob."""
    llm = LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32")
    result = llm.generate(
        messages=[{"role": "user", "content": "Hi"}],
        max_new_tokens=2,
        output_scores=True,
    )
    assert result.debug_text is not None
    assert any("mean_logprob:" in d for d in result.debug_text)


def test_model_loaded_with_correct_dtype():
    """Verify the model was actually loaded with torch_dtype (not ignored 'dtype')."""
    llm = LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32")
    # The model parameters should be float32
    param = next(llm.model.parameters())
    assert param.dtype == torch.float32
