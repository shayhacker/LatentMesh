"""OOP latent-space workflow built on top of LangGraph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .model import LatentLLM


class LatentTransform(ABC):
    """Transformation contract for latent stages."""

    @abstractmethod
    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Return transformed latent vector."""


class IdentityTransform(LatentTransform):
    """No-op latent transform."""

    def forward(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(latent, dtype=np.float32)


@dataclass(slots=True)
class CallableTransform(LatentTransform):
    """Wraps a Python callable as a latent transform."""

    fn: Callable[[np.ndarray], np.ndarray]

    def forward(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(latent, dtype=np.float32)), dtype=np.float32)


@dataclass(slots=True)
class LinearTransform(LatentTransform):
    """Affine transform with optional activation."""

    weight: np.ndarray
    bias: np.ndarray | None = None
    activation: str = "identity"

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=np.float32)
        if self.weight.ndim != 2:
            raise ValueError("weight must be rank-2")
        if self.bias is not None:
            self.bias = np.asarray(self.bias, dtype=np.float32)
            if self.bias.ndim != 1 or self.bias.shape[0] != self.weight.shape[1]:
                raise ValueError("bias must have shape [output_width]")
        if self.activation not in {"identity", "tanh", "relu"}:
            raise ValueError("activation must be one of: identity, tanh, relu")

    def forward(self, latent: np.ndarray) -> np.ndarray:
        vector = np.asarray(latent, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("latent must be rank-1")
        if vector.shape[0] != self.weight.shape[0]:
            raise ValueError(
                f"input width mismatch: expected {self.weight.shape[0]}, got {vector.shape[0]}"
            )

        output = vector @ self.weight
        if self.bias is not None:
            output = output + self.bias
        if self.activation == "tanh":
            output = np.tanh(output)
        elif self.activation == "relu":
            output = np.maximum(0.0, output)
        return output.astype(np.float32)


@dataclass(slots=True)
class LatentStage:
    """Single latent stage in a workflow."""

    name: str
    transform: LatentTransform = field(default_factory=IdentityTransform)

    def run(self, latent: np.ndarray) -> np.ndarray:
        output = np.asarray(self.transform.forward(latent), dtype=np.float32)
        if output.ndim != 1:
            raise ValueError(f"stage '{self.name}' must return a rank-1 vector")
        if not np.isfinite(output).all():
            raise ValueError(f"stage '{self.name}' produced NaN/Inf values")
        return output


@dataclass(frozen=True, slots=True)
class LatentStepTrace:
    """Trace record for one stage execution."""

    stage: str
    input_norm: float
    output_norm: float


@dataclass(frozen=True, slots=True)
class LatentRunTrace:
    """Trace information returned by workflow runs."""

    steps: tuple[LatentStepTrace, ...]
    output_latent: np.ndarray
    retrieved_examples: tuple[tuple[str, str], ...]


@dataclass(slots=True)
class LatentExampleMemory:
    """Retrieval memory keyed by latent similarity."""

    max_examples: int = 1024
    _items: list[tuple[np.ndarray, str, str]] = field(default_factory=list)

    def add(self, llm: LatentLLM, question: str, answer: str) -> None:
        vector = np.asarray(llm.embed(question), dtype=np.float32)
        self._items.append((vector, question, answer))
        overflow = len(self._items) - self.max_examples
        if overflow > 0:
            del self._items[:overflow]

    def retrieve(self, probe: np.ndarray, k: int) -> tuple[tuple[str, str], ...]:
        if not self._items or k <= 0:
            return ()

        scored = sorted(
            self._items,
            key=lambda item: _cosine_similarity(item[0], probe),
            reverse=True,
        )
        return tuple((question, answer) for _, question, answer in scored[:k])


@dataclass(slots=True)
class LatentRunner:
    """Runnable wrapper compatible with LangChain/LangServe expectations."""

    workflow: "LatentWorkflow"

    def invoke(self, input: Any, config: dict[str, Any] | None = None, **kwargs: Any) -> str:
        question = _extract_question(input)
        max_new_tokens = _extract_int("max_new_tokens", input, config, kwargs, default=256)
        temperature = _extract_float("temperature", input, config, kwargs, default=0.2)
        return self.workflow.invoke(
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def batch(self, inputs: list[Any], config: dict[str, Any] | None = None, **kwargs: Any) -> list[str]:
        return [self.invoke(item, config=config, **kwargs) for item in inputs]


@dataclass(slots=True)
class LatentBuilder:
    """OOP builder for latent workflows compiled through LangGraph."""

    llm: LatentLLM
    name: str = "latent-workflow"
    retrieval_k: int = 3
    memory: LatentExampleMemory = field(default_factory=LatentExampleMemory)
    _stages: dict[str, LatentStage] = field(default_factory=dict, init=False)
    _edges: list[tuple[str, str]] = field(default_factory=list, init=False)

    def add_stage(
        self,
        stage_name: str,
        *,
        transform: LatentTransform | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "LatentBuilder":
        if stage_name in self._stages:
            raise ValueError(f"stage '{stage_name}' already exists")
        self._stages[stage_name] = LatentStage(name=stage_name, transform=_coerce_transform(transform))
        return self

    def connect(self, source_stage: str, target_stage: str) -> "LatentBuilder":
        if source_stage not in self._stages:
            raise ValueError(f"unknown source stage '{source_stage}'")
        if target_stage not in self._stages:
            raise ValueError(f"unknown target stage '{target_stage}'")
        self._edges.append((source_stage, target_stage))
        return self

    def add_example(self, question: str, answer: str) -> "LatentBuilder":
        self.memory.add(self.llm, question, answer)
        return self

    def compile(
        self,
        *,
        entry_stage: str,
        exit_stage: str,
        prefer_langgraph: bool = True,
    ) -> "LatentWorkflow":
        if not self._stages:
            raise ValueError("at least one stage is required")
        if entry_stage not in self._stages:
            raise ValueError(f"entry_stage '{entry_stage}' is not defined")
        if exit_stage not in self._stages:
            raise ValueError(f"exit_stage '{exit_stage}' is not defined")

        workflow = LatentWorkflow(
            llm=self.llm,
            name=self.name,
            stages=dict(self._stages),
            edges=tuple(self._edges),
            entry_stage=entry_stage,
            exit_stage=exit_stage,
            retrieval_k=self.retrieval_k,
            memory=self.memory,
            prefer_langgraph=prefer_langgraph,
        )
        return workflow


class LatentGraph(LatentBuilder):
    """Semantic alias for LangGraph-first workflow building."""


@dataclass(slots=True)
class LatentWorkflow:
    """Compiled workflow executed through LangGraph when available."""

    llm: LatentLLM
    name: str
    stages: dict[str, LatentStage]
    edges: tuple[tuple[str, str], ...]
    entry_stage: str
    exit_stage: str
    retrieval_k: int
    memory: LatentExampleMemory
    prefer_langgraph: bool = True
    _topological_order: tuple[str, ...] = field(init=False, repr=False)
    _predecessors: dict[str, tuple[str, ...]] = field(init=False, repr=False)
    _compiled_graph: Any = field(init=False, default=None, repr=False)
    _uses_langgraph: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._topological_order = tuple(_topological_sort(self.stages, self.edges))
        self._predecessors = {
            stage_name: tuple(source for source, target in self.edges if target == stage_name)
            for stage_name in self.stages
        }
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        if not self.prefer_langgraph:
            self._uses_langgraph = False
            self._compiled_graph = _FallbackCompiledGraph(self)
            return

        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            self._uses_langgraph = False
            self._compiled_graph = _FallbackCompiledGraph(self)
            return

        graph = StateGraph(dict)
        graph.add_node("_ingress", self._ingress_step)

        for stage_name in self._topological_order:
            graph.add_node(stage_name, self._build_stage_step(stage_name))

        graph.add_node("_egress", self._egress_step)

        graph.add_edge(START, "_ingress")
        graph.add_edge("_ingress", self._topological_order[0])

        for index in range(len(self._topological_order) - 1):
            graph.add_edge(self._topological_order[index], self._topological_order[index + 1])

        graph.add_edge(self._topological_order[-1], "_egress")
        graph.add_edge("_egress", END)

        self._compiled_graph = graph.compile()
        self._uses_langgraph = True

    @property
    def uses_langgraph(self) -> bool:
        """True when running on a compiled LangGraph app."""

        return self._uses_langgraph

    @property
    def compiled_graph(self) -> Any:
        """Underlying compiled graph object (LangGraph or fallback executor)."""

        return self._compiled_graph

    def add_example(self, question: str, answer: str) -> None:
        self.memory.add(self.llm, question, answer)

    def invoke(
        self,
        question: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        answer, _ = self.run(
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return answer

    def batch(
        self,
        questions: list[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> list[str]:
        return [
            self.invoke(question, max_new_tokens=max_new_tokens, temperature=temperature)
            for question in questions
        ]

    def run(
        self,
        question: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> tuple[str, LatentRunTrace]:
        if not question.strip():
            raise ValueError("question must be non-empty")

        state = {
            "question": question,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "trace": [],
            "stage_outputs": {},
        }
        result = self._compiled_graph.invoke(state)

        answer = str(result["answer"])
        final_latent = np.asarray(result["final_latent"], dtype=np.float32)
        steps = tuple(
            LatentStepTrace(
                stage=str(entry["stage"]),
                input_norm=float(entry["input_norm"]),
                output_norm=float(entry["output_norm"]),
            )
            for entry in result.get("trace", [])
        )
        retrieved_examples = tuple(
            (str(question_text), str(answer_text))
            for question_text, answer_text in result.get("retrieved_examples", ())
        )

        return answer, LatentRunTrace(
            steps=steps,
            output_latent=final_latent,
            retrieved_examples=retrieved_examples,
        )

    def as_runnable(self) -> Any:
        """Return a LangChain RunnableLambda when available."""

        wrapper = LatentRunner(self)
        try:
            from langchain_core.runnables import RunnableLambda
        except Exception:
            return wrapper
        return RunnableLambda(lambda payload: wrapper.invoke(payload))

    def _ingress_step(self, state: dict[str, Any]) -> dict[str, Any]:
        question = str(state["question"])
        latent = np.asarray(self.llm.embed(question), dtype=np.float32)
        if latent.ndim != 1:
            raise ValueError("model embed returned non rank-1 vector")
        return {"initial_latent": latent, "trace": list(state.get("trace", [])), "stage_outputs": {}}

    def _build_stage_step(self, stage_name: str):
        def _stage_step(state: dict[str, Any]) -> dict[str, Any]:
            stage = self.stages[stage_name]
            stage_outputs = dict(state.get("stage_outputs", {}))
            predecessor_names = self._predecessors[stage_name]

            if predecessor_names:
                incoming = [np.asarray(stage_outputs[name], dtype=np.float32) for name in predecessor_names]
            else:
                incoming = [np.asarray(state["initial_latent"], dtype=np.float32)]

            try:
                merged = np.mean(np.stack(incoming, axis=0), axis=0)
            except ValueError as exc:
                raise ValueError(
                    f"stage '{stage_name}' received incompatible latent widths from predecessors"
                ) from exc

            output = stage.run(merged)
            stage_outputs[stage_name] = output

            trace = list(state.get("trace", []))
            trace.append(
                {
                    "stage": stage_name,
                    "input_norm": float(np.linalg.norm(merged)),
                    "output_norm": float(np.linalg.norm(output)),
                }
            )
            return {"stage_outputs": stage_outputs, "trace": trace, "latent": output}

        return _stage_step

    def _egress_step(self, state: dict[str, Any]) -> dict[str, Any]:
        stage_outputs = dict(state.get("stage_outputs", {}))
        if self.exit_stage not in stage_outputs:
            raise RuntimeError(f"exit stage '{self.exit_stage}' did not produce output")

        final_latent = np.asarray(stage_outputs[self.exit_stage], dtype=np.float32)
        question = str(state["question"])
        examples = self.memory.retrieve(final_latent, self.retrieval_k)
        prompt = self._build_prompt(question, examples)

        answer = self.llm.generate(
            prompt,
            max_new_tokens=int(state.get("max_new_tokens", 256)),
            temperature=float(state.get("temperature", 0.2)),
        )

        return {
            "answer": answer,
            "retrieved_examples": examples,
            "final_latent": final_latent,
        }

    @staticmethod
    def _build_prompt(question: str, examples: tuple[tuple[str, str], ...]) -> str:
        if not examples:
            return f"Question: {question}\nAnswer:"

        lines = ["Use examples as guidance.", "Examples:"]
        for sample_question, sample_answer in examples:
            lines.append(f"Q: {sample_question}")
            lines.append(f"A: {sample_answer}")
        lines.append("")
        lines.append(f"Question: {question}")
        lines.append("Answer:")
        return "\n".join(lines)


@dataclass(slots=True)
class _FallbackCompiledGraph:
    """Small fallback runtime used when LangGraph is not installed."""

    workflow: LatentWorkflow

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        current = dict(state)
        current.update(self.workflow._ingress_step(current))
        for stage_name in self.workflow._topological_order:
            current.update(self.workflow._build_stage_step(stage_name)(current))
        current.update(self.workflow._egress_step(current))
        return current


def _coerce_transform(
    transform: LatentTransform | Callable[[np.ndarray], np.ndarray] | None,
) -> LatentTransform:
    if transform is None:
        return IdentityTransform()
    if isinstance(transform, LatentTransform):
        return transform
    if callable(transform):
        return CallableTransform(transform)
    raise TypeError("transform must be a LatentTransform, callable, or None")


def _topological_sort(
    stages: dict[str, LatentStage],
    edges: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> list[str]:
    indegree = {stage_name: 0 for stage_name in stages}
    outgoing: dict[str, list[str]] = {stage_name: [] for stage_name in stages}

    for source_stage, target_stage in edges:
        if source_stage not in stages:
            raise ValueError(f"edge source '{source_stage}' is not a defined stage")
        if target_stage not in stages:
            raise ValueError(f"edge target '{target_stage}' is not a defined stage")
        indegree[target_stage] += 1
        outgoing[source_stage].append(target_stage)

    queue = sorted([stage_name for stage_name, degree in indegree.items() if degree == 0])
    order: list[str] = []

    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(order) != len(stages):
        raise ValueError("workflow graph must be a DAG (cycle detected)")

    return order


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)

    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def _extract_question(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, dict):
        for key in ("input", "question", "query", "text"):
            if key in input_value:
                return str(input_value[key])
    if hasattr(input_value, "to_string"):
        return str(input_value.to_string())
    return str(input_value)


def _extract_int(
    name: str,
    input_value: Any,
    config: dict[str, Any] | None,
    kwargs: dict[str, Any],
    *,
    default: int,
) -> int:
    if name in kwargs:
        return int(kwargs[name])
    if isinstance(input_value, dict) and name in input_value:
        return int(input_value[name])
    if isinstance(config, dict) and name in config:
        return int(config[name])
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    if name in configurable:
        return int(configurable[name])
    return default


def _extract_float(
    name: str,
    input_value: Any,
    config: dict[str, Any] | None,
    kwargs: dict[str, Any],
    *,
    default: float,
) -> float:
    if name in kwargs:
        return float(kwargs[name])
    if isinstance(input_value, dict) and name in input_value:
        return float(input_value[name])
    if isinstance(config, dict) and name in config:
        return float(config[name])
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    if name in configurable:
        return float(configurable[name])
    return default


__all__ = [
    "LatentTransform",
    "IdentityTransform",
    "CallableTransform",
    "LinearTransform",
    "LatentStage",
    "LatentStepTrace",
    "LatentRunTrace",
    "LatentExampleMemory",
    "LatentRunner",
    "LatentBuilder",
    "LatentGraph",
    "LatentWorkflow",
]
