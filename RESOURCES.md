# RESOURCES

Generated: 2026-03-22

This document covers only Step 1:
1. Read each requested paper.
2. Clone and inspect each listed GitHub repo (or record access failure).
3. Synthesize concrete product direction for approval before implementation.

---

## 0) Source Map

| ID | Paper | Code | Status |
|---|---|---|---|
| 2510.03215 | Cache-to-Cache: Direct Semantic Communication Between Large Language Models | https://github.com/thu-nics/C2C | Paper read + repo inspected |
| 2510.20733 | Thought Communication in Multiagent Collaboration | - | Paper read |
| 2511.09149 | Enabling Agents to Communicate Entirely in Latent Space | - | Paper read |
| 2511.20639 | Latent Collaboration in Multi-Agent Systems | https://github.com/Gen-Verse/LatentMAS | Paper read + repo inspected |
| 2512.20629 | Learning Evolving Latent Strategies for Multi-Agent Language Systems without Model Fine-Tuning | - | Paper read |
| 2601.06123 | Latent Space Communication via K-V Cache Alignment | - | Paper read |
| 2602.00471 | Dual Latent Memory for Visual Multi-agent System | https://github.com/YU-deep/L2-VMAS | Paper read; repo inaccessible (404/auth) |
| 2602.03026 | Visual Reasoning over Time Series via Multi-Agent System | - | Paper read |
| 2602.03036 | LatentMem: Customizing Latent Memory for Multi-Agent Systems | https://github.com/KANABOON1/LatentMem | Paper read + repo inspected |
| 2602.03695 | Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems | - | Paper read |
| 2602.15382 | The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems | https://github.com/xz-liu/heterogeneous-latent-mas | Paper read + repo inspected |

---

## 1) Paper Docs

### 1.1 arXiv:2510.03215 - Cache-to-Cache (C2C)
- Links:
  - Paper: https://arxiv.org/abs/2510.03215
  - Code: https://github.com/thu-nics/C2C
- Problem:
  - Text-mediated inter-agent communication is low-bandwidth, ambiguous, and slow.
- Core method:
  - Direct sharer-to-receiver KV-cache projection and fusion.
  - Learnable fuser/projector with layer selection (gating), token/layer alignment between heterogeneous models.
  - Oracle studies motivate both cache enrichment and cross-model cache convertibility.
- Key empirical claims:
  - Better than single-model and text-based multi-model communication.
  - Reported gains roughly: +6.4% to +14.2% over single models, +3.1% to +5.4% over text communication, ~2.5x latency speedup.
- Practical implications:
  - Strong foundation for low-latency latent transport between LLMs.
  - Layer-wise projection abstractions are directly reusable in a product runtime.
- Risks/gaps:
  - Pairwise projector training can become costly at scale.
  - Implementation heavily depends on cache internals and model-specific behavior.

### 1.2 arXiv:2510.20733 - Thought Communication in Multiagent Collaboration
- Link: https://arxiv.org/abs/2510.20733
- Problem:
  - Token communication underuses internal thought structure.
- Core method:
  - Formal latent-variable treatment of shared/private thoughts across agents.
  - Theoretical identifiability results for thought factors and sharing structure.
  - THOUGHTCOMM framework: latent extraction, structure-aware assignment, prefix-style latent injection.
- Key empirical claims:
  - Synthetic experiments validate identifiability.
  - Real benchmark improvements on MATH/GSM8K versus single-answer and multi-agent finetuning baselines.
  - Example reported result: Qwen3-1.7B on MATH up to ~93% with notable margin over baseline setup.
- Practical implications:
  - Provides theoretical backing for learning shared latent schemas.
  - Useful for designing universal latent routing metadata.
- Risks/gaps:
  - No reference implementation provided in requested resources.
  - Finetuning/training components are non-trivial for production hardening.

### 1.3 arXiv:2511.09149 - Interlat
- Link: https://arxiv.org/abs/2511.09149
- Problem:
  - Natural-language communication is lossy and expensive in multi-agent collaboration.
- Core method:
  - Inter-agent latent communication using last hidden states.
  - Additional learned compression stage for shorter latent messages.
  - Objectives include predictive preservation and latent geometry alignment.
- Key empirical claims:
  - Strong task performance on ALFWorld and robust behavior under perturbation/cross-family settings.
  - Compression can drastically reduce communication latency (reported up to ~24x) while preserving useful performance.
- Practical implications:
  - Strong evidence that latent compression can be a first-class runtime primitive.
- Risks/gaps:
  - Training pipeline complexity and dependency on setup choices.
  - No requested code repo here for direct reproducibility.

### 1.4 arXiv:2511.20639 - LatentMAS
- Links:
  - Paper: https://arxiv.org/abs/2511.20639
  - Code: https://github.com/Gen-Verse/LatentMAS
- Problem:
  - Need end-to-end latent collaboration framework that avoids text bottlenecks.
- Core method:
  - Training-free latent thought rollout inside each agent.
  - Shared latent working memory via KV cache transfer across agents.
  - Optional latent-space realignment for hidden-to-embedding consistency.
- Key empirical claims:
  - Up to ~14.6% accuracy gains, ~70.8%-83.7% token reduction, and ~4x to 4.3x end-to-end speedup versus text MAS baselines in reported settings.
- Practical implications:
  - Good template for near-zero-training deployment mode.
  - Good benchmark coverage and practical scripts.
- Risks/gaps:
  - Current implementation still ends with text decoding for final answers.
  - vLLM path relies on backend modifications; not cleanly upstream-safe.

### 1.5 arXiv:2512.20629 - Evolving Latent Strategies Without Fine-Tuning
- Link: https://arxiv.org/abs/2512.20629
- Problem:
  - How to evolve strategies over time without modifying LLM weights.
- Core method:
  - Dual-loop architecture:
    - Behavior loop: Q-learning updates.
    - Language loop: reflection embeddings update external latent strategy vectors.
  - Multi-role agent setup with trust-weighted meta-controller.
- Key empirical claims:
  - Latent strategy trajectories converge and become role-distinct.
  - Emergent adaptation phenomena observed in their grid-style environment.
- Practical implications:
  - Useful for low-cost online adaptation where model weights must stay frozen.
- Risks/gaps:
  - Experimental environment appears narrow/toy relative to enterprise tasks.
  - Limited evidence on large-scale real workloads.

### 1.6 arXiv:2601.06123 - Latent Space Communication via K-V Cache Alignment
- Link: https://arxiv.org/abs/2601.06123
- Problem:
  - Need scalable cross-model latent interoperability.
- Core method:
  - Learn global shared KV latent space plus per-model adapters into/out of shared space.
  - Keep base models frozen; train only translators.
  - Supports model pools with different sizes, seeds, and language specializations.
- Key empirical findings:
  - Prefix-cache translation can improve suffix modeling in several settings.
  - Demonstrated portability of learned modules (for example, soft prompts) via shared latent space.
  - Demonstrated extensibility: add new model adapters without retraining all existing ones.
- Practical implications:
  - Very relevant for O(N) onboarding design versus O(N^2) pairwise translators.
- Risks/gaps:
  - Results centered on Gemma-scale experiments and language modeling style evaluation.
  - Product-grade routing/reliability layers are not included.

### 1.7 arXiv:2602.00471 - L2-VMAS
- Links:
  - Paper: https://arxiv.org/abs/2602.00471
  - Code URL in paper: https://github.com/YU-deep/L2-VMAS
- Problem:
  - Visual MAS "scaling wall": more turns can hurt quality while exploding token cost.
- Core method:
  - Dual latent memory:
    - Perception memory (multi-granularity visual latent units).
    - Thinking memory (entropy-aware chunked latent thought units).
  - Entropy-driven proactive retrieval and dynamic memory management.
- Key empirical claims:
  - Reported average +2.7% to +5.4% accuracy gains and -21.3% to -44.8% token usage versus VMAS baselines.
- Practical implications:
  - Strong memory orchestration ideas for multimodal latent systems.
- Risks/gaps:
  - Repo inaccessible at time of audit (details in repo section), limiting reproducibility.

### 1.8 arXiv:2602.03026 - MAS4TS
- Link: https://arxiv.org/abs/2602.03026
- Problem:
  - General time-series tasks need stronger visual reasoning + tool routing.
- Core method:
  - Analyzer-Reasoner-Executor multi-agent pattern.
  - Visual reasoning over plotted series, latent reconstruction, and tool-chain routing.
- Key empirical claims:
  - Strong benchmark outcomes across forecasting, classification, imputation, and anomaly detection.
- Practical implications:
  - Valuable for verticalized latent-agent applications (time series, operational analytics).
- Risks/gaps:
  - Not a pure latent inter-agent communication paper; more domain workflow oriented.

### 1.9 arXiv:2602.03036 - LatentMem
- Links:
  - Paper: https://arxiv.org/abs/2602.03036
  - Code: https://github.com/KANABOON1/LatentMem
- Problem:
  - Existing MAS memory is often role-agnostic and token-heavy.
- Core method:
  - Experience bank stores raw trajectories.
  - Memory composer synthesizes role-aware compact latent memories.
  - Latent Memory Policy Optimization (LMPO) backpropagates task signal through latent memories.
- Key empirical claims:
  - Up to ~19.36% performance improvement over vanilla settings.
  - Strong gains across in-domain, out-of-domain, and unseen MAS settings.
  - Roughly half token usage and notable inference-time reduction in reported setup.
- Practical implications:
  - Strong blueprint for latent memory subsystem in production.
- Risks/gaps:
  - Training/deployment stack is heavy.
  - Repo currently lacks explicit license file.

### 1.10 arXiv:2602.03695 - Agent Primitives
- Link: https://arxiv.org/abs/2602.03695
- Problem:
  - Existing MAS architectures are task-specific and hard to reuse.
- Core method:
  - Reusable primitive blocks:
    - Review
    - Voting and Selection
    - Planning and Execution
  - KV-cache latent communication inside primitives.
  - Organizer agent composes primitives using a lightweight knowledge pool.
- Key empirical claims:
  - Reported +12.0% to +16.5% average accuracy over single-agent.
  - ~3x to 4x lower tokens/latency than text-MAS baselines.
  - ~1.3x to 1.6x overhead relative to single-agent.
- Practical implications:
  - Very strong productizable abstraction for plug-and-play architecture composition.
- Risks/gaps:
  - No requested official implementation link in resource list.

### 1.11 arXiv:2602.15382 - Vision Wormhole
- Links:
  - Paper: https://arxiv.org/abs/2602.15382
  - Code: https://github.com/xz-liu/heterogeneous-latent-mas
- Problem:
  - Heterogeneous-model latent communication is hard with pairwise translators.
- Core method:
  - Universal Visual Codec and vision-token-span injection.
  - Hub-and-spoke alignment in universal token space to reduce scaling from O(N^2) to O(N).
  - Teacher-student objective aligns fast visual latent channel with text pathway behavior.
- Key empirical claims:
  - Main setting: macro +6.3pp accuracy and ~1.87x speedup versus text baseline (reported).
  - Weakly supervised codec: macro +6.5pp and ~2.67x speedup (reported).
- Practical implications:
  - Best available path for heterogeneous VLM latent exchange without per-pair explosion.
- Risks/gaps:
  - Paper and repo marked as work in progress.
  - Repo currently lacks explicit license file.

---

## 2) Repository Inspections

### 2.1 C2C repo audit
- URL: https://github.com/thu-nics/C2C
- Clone status: success.
- Snapshot info:
  - Branch: `main`
  - Latest commit seen: `113c3a9` (2026-02-06)
  - File count (tracked in clone): ~79
  - License: Apache-2.0 (`LICENSE` present)
- What is implemented:
  - `RosettaModel` wrapper for multi-model inference with KV projection/fusion.
  - Projector framework with configurable gating granularity and fusion behavior.
  - Multi-sharer bitmask routing, sequential/parallel fusion modes.
  - Token alignment utility for heterogeneous tokenizers.
  - Training/eval recipes and scripts; examples for chat and demo.
- Production-readiness observations:
  - Positive:
    - Clear modular decomposition.
    - Config-driven workflows.
    - Packaging via `pyproject.toml`.
  - Risks:
    - Qwen-specific monkeypatch path in wrapper internals is brittle.
    - Minimal automated testing footprint.
    - Environment files include heavy and platform-specific assumptions.

### 2.2 LatentMAS repo audit
- URL: https://github.com/Gen-Verse/LatentMAS
- Clone status: success.
- Snapshot info:
  - Branch: `main`
  - Latest commit seen: `c14da9c` (2026-02-27)
  - File count: ~23
  - License: Apache-2.0 (`LICENSE` present)
- What is implemented:
  - Baseline/TextMAS/LatentMAS runners for multiple tasks.
  - HF path and optional vLLM hybrid path.
  - Latent rollout by feeding aligned hidden vectors as embeddings.
  - Optional latent-space realignment matrix built from embedding/lm-head relation.
- Production-readiness observations:
  - Positive:
    - Lightweight and easy to read.
    - Straightforward CLI usage.
  - Risks:
    - Not packaged as a robust library.
    - No proper test suite.
    - vLLM route depends on local backend modifications and strict assumptions.

### 2.3 LatentMem repo audit
- URL: https://github.com/KANABOON1/LatentMem
- Clone status: success.
- Snapshot info:
  - Branch: `main`
  - Latest commit seen: `7173f64` (2026-02-09)
  - File count: ~89
  - License: no explicit license file found.
- What is implemented:
  - Full latent memory stack (experience bank + composer + runner + trainers).
  - LMPO/GRPO style training components.
  - Integrations with multiple MAS structures and datasets.
- Production-readiness observations:
  - Positive:
    - Rich functionality and broader scope than many peers.
    - Clear concept separation (memory/core/trainer/data).
  - Risks:
    - Heavy infra and dependency footprint (CUDA-heavy env, deepspeed, etc).
    - Missing license blocks direct OSS reuse in a product without clarification.

### 2.4 Heterogeneous-Vision Wormhole repo audit
- URL: https://github.com/xz-liu/heterogeneous-latent-mas
- Clone status: success.
- Snapshot info:
  - Branch: `main`
  - Latest commit seen: `f55e921` (2026-03-03)
  - File count: ~23
  - License: no explicit license file found.
- What is implemented:
  - Codec training pipeline for VLM latent-to-vision bridge.
  - Checkpoint merge tool for multi-family codec alignment.
  - Partitioned run scripts for large benchmark sweeps.
  - Additional baselines including OCR path.
- Production-readiness observations:
  - Positive:
    - Directly targets heterogeneous model interoperability.
    - Includes practical orchestration scripts for large runs.
  - Risks:
    - Repo explicitly labeled ongoing/WIP.
    - Significant model-specific compatibility patches.
    - Missing license.

### 2.5 L2-VMAS repo access check
- URL: https://github.com/YU-deep/L2-VMAS
- Clone status: failed.
- Observed behavior:
  - `git clone` returns auth prompt failure.
  - HTTP check returns 404 as of 2026-03-22.
- Conclusion:
  - Paper insights usable; direct code inspection not possible at this time.

---

## 3) Cross-Resource Synthesis (What actually matters for a product)

### 3.1 Converging design patterns
- Nearly all high-performing latent-MAS systems converge on the same core loop:
  - local latent thought rollout
  - compact latent message construction
  - cross-agent latent injection
  - optional memory persistence/retrieval
- Two scalability strategies dominate:
  - Shared latent space adapters (C2C/KV alignment style).
  - Hub-and-spoke universal codec (Vision Wormhole style).
- Latent memory is increasingly central (LatentMem, L2-VMAS):
  - Without memory management, quality and efficiency degrade as turns grow.

### 3.2 Non-negotiable product requirements inferred from resources
- Strict separation of communication plane and decode plane.
- Explicit universal latent packet schema (shape, dtype, layer mapping, provenance).
- O(N) onboarding for models (avoid pairwise O(N^2) translators).
- Pluggable memory with token-efficient latent summaries.
- First-class observability of latent pipeline health (shape drift, norm drift, entropy, alignment score).

### 3.3 Main engineering risks
- Tight coupling to model internals (cache formats, attention implementations).
- Unstable vLLM internals when modifying prompt-embed/KV behavior.
- License risk from missing-license repos if code is directly reused.
- Evaluation mismatch: many papers benchmark QA/code tasks but not real tool-heavy enterprise workflows.

---

## 4) Proposed Product (for approval before Step 2)

## Product name (working): **LatentMesh**

A production-first latent communication runtime for multi-agent systems where inter-agent exchange is strictly latent (no inter-agent text decode path).

### 4.1 Product thesis
- Existing MAS frameworks are easy to prototype but expensive/fragile at scale due to text communication.
- We can provide an OSS runtime that upgrades any agent stack with a latent transport layer, model adapters, and latent memory, while preserving existing orchestration APIs.

### 4.2 What LatentMesh should include in v1
- **Latent Transport Core**
  - Universal latent packet abstraction.
  - Zero-text inter-agent communication channel.
  - Strict mode guardrails that reject token-decoding in communication paths.
- **Adapter SDK**
  - KV/cache-based adapters (C2C/KV-alignment inspired).
  - Vision-token codec adapters (Vision Wormhole inspired) for heterogeneous VLMs.
  - O(N) hub adapters rather than pairwise mesh.
- **Latent Memory Engine**
  - Experience bank + latent composer (LatentMem inspired).
  - Retrieval + role-aware memory projection.
- **Serving Runtime**
  - vLLM-compatible fast path (forked where required).
  - Batched latent routing, async execution, backpressure controls.
- **Framework Connectors**
  - LangChain / LlamaIndex / custom Python agent loop adapters.
  - "Drop-in" wrappers around existing agent roles.
- **Eval + Observability**
  - Bench harness for quality/latency/token-cost deltas.
  - Latent telemetry: norm drift, alignment confidence, packet entropy, failure traces.

### 4.3 Why this is useful (not a toy)
- Cuts communication latency and token spend in multi-agent pipelines.
- Enables heterogeneous model collaboration without building pairwise translators per model pair.
- Adds memory quality controls that are absent from most text-MAS products.
- Gives a practical path for enterprise inference cost reduction while improving quality on hard tasks.

### 4.4 Strict "no text/tokens generated" interpretation
- In strict mode:
  - Inter-agent communication is latent-only.
  - No natural-language intermediate messages are generated between agents.
  - Communication/logging paths use latent telemetry, not textual thought traces.
- Note:
  - External user-facing output mode can be configured separately; strict latent system mode will keep all intra-system collaboration latent-only.

### 4.5 Scope recommendation for Step 2 implementation
- Build a complete OSS skeleton + working end-to-end runtime with:
  - one text-model adapter path,
  - one heterogeneous adapter path,
  - latent memory,
  - framework connectors,
  - local benchmark suite,
  - production docs and examples.
- Keep optional advanced research variants behind feature flags.

---

## 5) Suggested "Do Next" after approval
- If approved, Step 2 should implement LatentMesh with a production codebase structure:
  - `latentmesh-core` (runtime, packet protocol, adapters)
  - `latentmesh-serve` (high-performance inference/queueing)
  - `latentmesh-connectors` (LangChain/LlamaIndex/etc.)
  - `latentmesh-evals` (reproducible benchmarks)
  - full docs focused on usage and deployment.

