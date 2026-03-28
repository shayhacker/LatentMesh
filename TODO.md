# TODO

## Performance
- [ ] **Parallelise VotingPrimitive candidates.** Currently runs each candidate sequentially on the same model. Could use `asyncio` or thread-pool if model supports concurrent inference, or batch prompts together.
- [ ] **GlobalPrefixCache garbage collection.** The trie grows unboundedly. Add TTL or LRU eviction for long-running sessions.
- [ ] **Streaming generation.** Support `yield`-based token streaming through LangGraph nodes for real-time output.

## Features
- [ ] **CI/CD pipeline.** Add GitHub Actions: lint (ruff), type check (mypy), test runner, benchmark smoke test.
- [ ] **CHANGELOG.md.** Track releases with semantic versioning and git tags.
- [ ] **Custom primitive tutorial.** Document how to subclass `AgentPrimitive` with custom trigger logic.
- [ ] **Multi-model architectures guide.** Document using different models for different agents.
- [ ] **DiskKVStore integration test.** Add test that exercises `diskcache`-backed persistence across process restarts.

## Code Quality
- [ ] **Type annotations.** Add `py.typed` marker and improve type coverage for mypy strict mode.
- [ ] **Thread safety audit.** `MemoryKVStore` is not thread-safe. Add locking or document single-thread constraint.
