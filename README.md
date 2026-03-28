<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![License][license-shield]][license-url]
[![Python][python-shield]][python-url]

</div>

<!-- PROJECT HEADER -->
<br />
<div align="center">
  <h3 align="center">LatentMesh</h3>

  <p align="center">
    Multi-agent Latent Space Communication for LLMs
    <br />
    <br />
    <a href="latentmesh.vercel.app"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/shayhacker/LatentMesh/issues/new?labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/shayhacker/LatentMesh/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

LatentMesh wires multiple LLM agents together in a [LangGraph](https://github.com/langchain-ai/langgraph) pipeline by strictly communicating via KV cache.

In standard architectures, when Agent B needs to read what Agent A computed, it must process exactly the same text history from scratch. With LatentMesh, Agent B can access the prior agent's KV cache states, cutting costs computation costs from $O(n^2)$ to $O(n)$. Our long term vision is to expand communication between any model regardless of differences in architecture.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation

Clone the repository and install in editable mode

```sh
git clone https://github.com/shayhacker/LatentMesh.git
cd LatentMesh
pip install -e .
```
  
Alternatively, install via PIP
> not yet available

```sh
pip install latentmesh
```

**Optional Dependencies:**
* For persistent disk-backed caching (recommended for production):
  ```sh
  pip install latentmesh[disk]
  ```
* For booting a ready-to-go FastAPI endpoint:
  ```sh
  pip install latentmesh[server]
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

A basic multi-agent configuration building a `Plan -> Reason -> Review` text pipeline.

```python
from langgraph.graph import StateGraph, START, END
from latentmesh import LatentLLM, LatentState
from latentmesh.primitives import PlanPrimitive, ReasonPrimitive, ReviewPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache

# 1. Provide the cache to skip re-encoding string prefixes
store = MemoryKVStore()
cache = GlobalPrefixCache(store)

# 2. Use a standard HF model
llm = LatentLLM("Qwen/Qwen3-0.6B", device="cuda", global_cache=cache)

# 3. Create discrete agent primitives
planner  = PlanPrimitive(llm)
reasoner = ReasonPrimitive(llm)
reviewer = ReviewPrimitive(llm)

# 4. Connect via LangGraph 
builder = StateGraph(LatentState)
builder.add_node("planner", planner)
builder.add_node("reasoner", reasoner)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "planner")
builder.add_edge("planner", "reasoner")
builder.add_edge("reasoner", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()

# 5. Run it forward sequentially
result = graph.invoke({
    "messages": [{"role": "user", "content": "What is the cosine of 45 degrees?"}],
    "tokens_so_far": 0,
})

print(result["latent"].text)
print(f"Total tokens generated: {result['tokens_so_far']}")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXAMPLES -->
## Examples

For more involved architectures like consensus routing and dynamic state inspection, check out `examples/`:

| Example | Description |
|---|---|
| [`sequential.py`](examples/sequential.py) | Plan → Reason → Review zero-shot pipeline |
| [`complex.py`](examples/complex.py) | Fast consensus parallel-voting with `VotingPrimitive` |
| [`hierarchical.py`](examples/hierarchical.py) | Dynamic routing based on model confidence scoring |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Implement HuggingFace model backend wrapping
- [x] Add global prefix matching via `pygtrie`
- [x] Create Memory and Disk KV Stores
- [ ] Multi-GPU parallelization for `VotingPrimitive`
- [ ] Implement LRU/TTL eviction policies for the global prefix cache
- [ ] Publish v1.0.0 to PyPI

See the [open issues](https://github.com/shayhacker/LatentMesh/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

* Yash Ranjith - yranjith@stanford.edu
* Hiroki Kimiwada - kimiwada@stanford.edu
* William Peng - wgpeng@stanford.edu
* Atharva Rao - arao2007@stanford.edu

Project Link: [https://github.com/shayhacker/LatentMesh](https://github.com/shayhacker/LatentMesh)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/shayhacker/LatentMesh.svg?style=for-the-badge
[contributors-url]: https://github.com/shayhacker/LatentMesh/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shayhacker/LatentMesh.svg?style=for-the-badge
[forks-url]: https://github.com/shayhacker/LatentMesh/network/members
[stars-shield]: https://img.shields.io/github/stars/shayhacker/LatentMesh.svg?style=for-the-badge
[stars-url]: https://github.com/shayhacker/LatentMesh/stargazers
[license-shield]: https://img.shields.io/github/license/shayhacker/LatentMesh.svg?style=for-the-badge
[license-url]: https://github.com/shayhacker/LatentMesh/blob/main/LICENSE
[python-shield]: https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[transformers-shield]: https://img.shields.io/badge/Transformers-FFCA28?style=for-the-badge&logo=huggingface&logoColor=black
[transformers-url]: https://huggingface.co/docs/transformers
[langgraph-shield]: https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white
[langgraph-url]: https://www.langchain.com/langgraph
[fastapi-shield]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
