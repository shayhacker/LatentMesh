# Development Status

Current project version: **v0.5.2**

---

## Completed

### Core Package (`latentmesh/`)
- `LatentLLM` with HuggingFace Transformers backend and `GlobalPrefixCache` integration
- `AgentOutput` dataclass with token tracking (cached, uncached, output)
- `extract_kv` format normaliser for cross-model KV cache compatibility
- `LatentState` LangGraph-compatible TypedDict with custom reducer
- `AgentPrimitive` base class with trigger-text steering
- `PlanPrimitive`, `ReasonPrimitive`, `ReviewPrimitive`
- `VotingPrimitive` with mean-logprob candidate selection
- `GlobalPrefixCache` with `pygtrie.CharTrie` for O(L) longest-prefix lookup
- `MemoryKVStore` for testing, `DiskKVStore` (via `diskcache`) for production
- Configurable `debug` flag for logging cache hits/misses and token accounting

### Examples (`examples/`)
- `sequential.py` — linear Plan → Reason → Review pipeline
- `complex.py` — multi-path VotingPrimitive consensus
- `hierarchical.py` — supervisor/worker routing via text content

### Tests (`tests/`)
- 27 tests across 5 files covering core generation, graph reducers, primitives, serialisation, and integration

### Documentation (`docs/`)
- Overview, architecture, deep-dive, API reference, and development status pages
- Vite-based documentation site

---

## Remaining

See [TODO.md](../TODO.md) for the full list.

### Priority Items
- CI/CD pipeline (GitHub Actions: lint, type check, tests)
- PyPI publishing (build and upload)
- CHANGELOG.md and release tags
- VotingPrimitive parallelisation
- GlobalPrefixCache TTL/eviction

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **v0.5** | Core API, examples, tests | ✅ Complete |
| **v0.6** | CI/CD, PyPI publish | 🔲 Next |
| **v0.7** | Parallel voting, cache eviction | 🔲 Planned |
| **v1.0** | Stable API, comprehensive docs | 🔲 Planned |
