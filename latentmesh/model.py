"""Model backends for latent-space workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np


class LatentModelBackend(ABC):
    """Backend contract used by `LatentLLM`."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Encode text to a rank-1 latent vector."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """Generate final user-facing text."""


class MockLatentBackend(LatentModelBackend):
    """Deterministic backend used for tests and local development."""

    def __init__(self, width: int = 256) -> None:
        self._width = int(width)

    def embed(self, text: str) -> np.ndarray:
        digest = sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._width).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-9
        return vector

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        last_line = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        if len(last_line) > 512:
            last_line = last_line[:512]
        del temperature
        return f"[mock-response] {last_line}"[: max(32, max_new_tokens)]


class TransformersLatentBackend(LatentModelBackend):
    """HuggingFace backend for open-source checkpoints."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Transformers backend requires 'transformers' and 'torch'. Install extras: .[transformers]"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id)
        self._model.eval()
        self._loaded = True

    def embed(self, text: str) -> np.ndarray:
        self._load()
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with self._torch.no_grad():
            outputs = self._model(**tokens, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1][0]
        vector = hidden.mean(dim=0)
        return vector.detach().cpu().numpy().astype(np.float32)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        self._load()
        tokens = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with self._torch.no_grad():
            outputs = self._model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(1e-4, temperature),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = outputs[0][tokens["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class OllamaLatentBackend(LatentModelBackend):
    """Ollama backend for local open-source models."""

    def __init__(self, model_id: str, embedding_model: str | None = None) -> None:
        self._model_id = model_id
        self._embedding_model = embedding_model or model_id

        try:
            import ollama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ollama backend requires 'ollama'. Install extras: .[ollama]") from exc

        self._ollama = ollama

    def embed(self, text: str) -> np.ndarray:
        response = self._ollama.embeddings(model=self._embedding_model, prompt=text)
        return np.asarray(response["embedding"], dtype=np.float32)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        response = self._ollama.generate(
            model=self._model_id,
            prompt=prompt,
            options={"num_predict": max_new_tokens, "temperature": temperature},
        )
        return str(response["response"]).strip()


@dataclass(slots=True)
class LatentLLM:
    """High-level model wrapper.

    Example:
      llm = LatentLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      latent = llm.embed("hello")
      answer = llm.generate("Explain recursion")
    """

    model_id: str
    backend: str = "auto"
    embedding_model: str | None = None
    mock_width: int = 256
    _backend_impl: LatentModelBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        backend = self.backend.lower().strip()

        if backend == "mock":
            self._backend_impl = MockLatentBackend(width=self.mock_width)
            return

        if backend == "ollama" or self.model_id.startswith("ollama:"):
            model_id = self.model_id.replace("ollama:", "", 1)
            self._backend_impl = OllamaLatentBackend(model_id=model_id, embedding_model=self.embedding_model)
            return

        if backend in {"auto", "hf", "transformers"}:
            self._backend_impl = TransformersLatentBackend(self.model_id)
            return

        raise ValueError("backend must be one of: auto, hf, transformers, ollama, mock")

    def embed(self, text: str) -> np.ndarray:
        vector = np.asarray(self._backend_impl.embed(text), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embed backend must return a rank-1 vector")
        if not np.isfinite(vector).all():
            raise ValueError("embed backend returned NaN/Inf")
        return vector

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        return self._backend_impl.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


__all__ = [
    "LatentModelBackend",
    "MockLatentBackend",
    "TransformersLatentBackend",
    "OllamaLatentBackend",
    "LatentLLM",
]
