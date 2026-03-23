from __future__ import annotations

import numpy as np
import pytest

from latentmesh import LatentLLM


def test_mock_embed_is_deterministic() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    first = llm.embed("hello")
    second = llm.embed("hello")
    assert first.ndim == 1
    assert np.allclose(first, second)


def test_mock_generate_returns_text() -> None:
    llm = LatentLLM("mock://dev", backend="mock")
    out = llm.generate("Question: test\nAnswer:")
    assert out.startswith("[mock-response]")


def test_invalid_backend_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        LatentLLM("x", backend="unknown")


def test_embed_validation_rank_1_enforced() -> None:
    llm = LatentLLM("mock://dev", backend="mock")

    class BadBackend:
        def embed(self, text: str):
            del text
            return np.zeros((2, 2), dtype=np.float32)

        def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
            del prompt, max_new_tokens, temperature
            return "ok"

    llm._backend_impl = BadBackend()
    with pytest.raises(ValueError, match="rank-1"):
        llm.embed("hello")


def test_embed_validation_finite_values_enforced() -> None:
    llm = LatentLLM("mock://dev", backend="mock")

    class BadBackend:
        def embed(self, text: str):
            del text
            return np.array([1.0, np.nan], dtype=np.float32)

        def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
            del prompt, max_new_tokens, temperature
            return "ok"

    llm._backend_impl = BadBackend()
    with pytest.raises(ValueError, match="NaN/Inf"):
        llm.embed("hello")
