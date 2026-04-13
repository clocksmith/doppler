# Doppler Architecture

## Architecture Overview

**Doppler** (Deterministic On-device Processing for Prefill, Learning, and Execution Runtime) is a standalone WebGPU-native ML runtime repo centered on forward inference and prefill in browser and Node environments.

The repo also contains experimental and internal-only subsystem surfaces for
training, distribution, hotswap, diffusion, energy, bridge integration, and
direct-source adaptation. The canonical subsystem support contract is generated
from `src/config/support-tiers/subsystems.json`; see
[subsystem-support-matrix.md](subsystem-support-matrix.md).

See also: [INDEX.md](INDEX.md)

## Overview

![Doppler architecture overview](architecture-overview.svg)

## Verification Sources

Use these code/doc locations to validate architecture claims in this file:

- Execution-plane contract and fail-fast policy:
  - `docs/style/general-style-guide.md`
  - `docs/style/command-interface-design-guide.md`
- Runtime config + kernel-path precedence:
  - `docs/config.md`
  - `src/inference/pipelines/text/model-load.js`
  - `src/config/merge.js`
- Manifest/build-time conversion behavior:
  - `src/converter/manifest-inference.js`
  - `src/converter/rope-config.js`
- RDRR shard/manifest layout:
  - `src/config/schema/manifest.schema.js`
  - `src/formats/rdrr/manifest.js`
  - `src/formats/rdrr/validation.js`
- Benchmark/claim methodology:
  - `docs/benchmark-methodology.md`
  - `benchmarks/vendors/README.md`

## Optional Ouroboros/Reploid Integration

Doppler can integrate with [Reploid](https://github.com/clocksmith/reploid) via the minimal Ouroboros substrate contract:
SharedArrayBuffer for coordination plus VFS file exchange for inference plans and
results. Integration notes are maintained in private wrapper docs.

## Design Philosophy

DOPPLER makes deliberate architectural tradeoffs that diverge from pre-compiled approaches (for example ONNX/TVM-style packaged graphs). The focus is explicit runtime control and auditable policy resolution in config/manifest contracts.

### Key Principles

| Principle | Implementation | Why |
|-----------|----------------|-----|
| **Code/Data Separation** | Generic WGSL kernels + weight shards | Enables shard verification and runtime adapter/component swaps |
| **GPU Fusion** | Most hot-path tensor ops stay on GPU | Keeps JS orchestration overhead secondary when GPU compute dominates (workload-dependent) |
| **Progressive Fusion** | Swap atomic kernels for fused kernels via config | Get the best of both worlds: hackability default, performance peaks |
| **Minimal Readback** | Logit readback is cadence-controlled | Avoids unnecessary GPU→CPU transfer overhead during decode |
| **JavaScript Orchestration** | JS dispatches GPU work, handles sampling | Debugging, rapid iteration, browser integration |

## Execution Planes

- **JSON plane** (`manifest.json`, checked-in config assets, rule maps): policy, selection, and execution contract.
- **JS plane** (`src/**/*.js` orchestration): merge/validate config, allocate buffers, build pipelines, dispatch work, collect artifacts.
- **WGSL plane** (`src/gpu/kernels/*.wgsl`): deterministic arithmetic execution only.

This section is a system overview. The normative execution-plane contract lives
in [`style/general-style-guide.md`](style/general-style-guide.md).

Behavior-changing choices must be fully represented before dispatch:

- No implicit runtime detection from model names at execution time.
- No hidden policy branching in WGSL.
- No implicit defaults for capability fallback; unresolved decisions fail fast.

### Doppler vs Transformers.js (v4): Execution Boundary View

This comparison is about architecture and control surfaces, not blanket speed claims.

| Axis | Doppler | Transformers.js (v4) |
|------|---------|----------------------|
| Model/runtime contract | Manifest + runtime config merged explicitly | ONNX model graph + runtime backend config |
| Kernel-path control | Explicit `kernelPath` in config | Kernel execution is selected through runtime/backend coverage |
| Fallback visibility | Config + trace categories map directly to execution choices | Requires graph/runtime-level inspection to attribute node placement |
| Iteration loop | Edit config or WGSL kernel and rerun | Export/runtime/backend alignment is often part of tuning |
| Auditability of claims | Compare command + normalized JSON + SVG map directly to config knobs | Same benchmark harness works, but knob-to-kernel mapping is less direct |

For public claims, tie statements to benchmark artifacts in `benchmarks/vendors/results/` and the exact compare command used.
Source anchors for this comparison:
- Doppler command/runtime contract: `src/tooling/command-api.js`, `src/inference/pipelines/text/model-load.js`
- Benchmark claim policy: `docs/benchmark-methodology.md`, `benchmarks/vendors/README.md`

### Progressive Fusion in Practice

Doppler begins with discrete WGSL operations (for example `gate_proj`, `up_proj`, `down_proj`) to maximize observability and debuggability. The architecture also supports progressively fused execution plans.

Through kernel-path and execution-plan selection (`runtime.inference.kernelPath` and the v1 execution graph), the runtime can switch from decomposed paths to fused paths without changing command semantics. See `src/inference/pipelines/text/model-load.js` and `src/config/kernel-path-loader.js`.

### GPU Execution Footprint

Profiled decode runs are often GPU-dominant, but exact ratios vary by model, kernel path,
runtime config, browser/driver, and hardware. Treat any percentage split as workload-specific.
For claimable numbers, use `docs/benchmark-methodology.md` and artifacts in
`benchmarks/vendors/results/`.

### Readback Minimization

| Operation | Location | Readback? |
|-----------|----------|-----------|
| Q/K/V projections | GPU matmul | No |
| Attention scores | GPU fused kernel | No |
| MoE router decisions | GPU softmax+topk | No |
| Expert FFN outputs | GPU matmul | No |
| Intermediate hidden states | GPU buffers | No |
| **Final logits** | **GPU → CPU** | **Yes (required for CPU-side sampling in current default flow)** |

Readback cadence is runtime-configurable (`runtime.inference.session.decodeLoop.readbackInterval`).
Default parity decode is typically per-token, but batched decode modes can amortize readbacks.
See `src/inference/pipelines/text/generator-steps.js`.

### Capability vs Performance Tradeoff

DOPPLER may trade some kernel-level peak performance for runtime composability and explicit policy control:

| Capability | Why this runtime favors explicit composition |
|------------|----------------------------------------------|
| Shard distribution | Artifact/shard transport is first-class in storage/distribution paths |
| LoRA hot-swap | Adapters are loaded/activated at runtime via adapter and hotswap surfaces |
| Multi-model orchestration | Runtime exposes explicit multi-model primitives and policy-layer hooks |

Non-core aspiration and roadmap items are tracked in private wrapper docs.

---

## Engine vs Orchestrator Boundary

DOPPLER is the **Engine** (mechanism); a caller/orchestrator is the **Policy Layer**. Reploid is one policy-layer implementation.

### The Principle

**DOPPLER never decides, it only executes.** All policy decisions (what to do, when to do it, what weights to use) come from the caller/orchestrator. DOPPLER provides primitives that accept parameters and execute them efficiently on the GPU.

In Doppler terms:

- **JSON contract** (`manifest`, config assets, rule assets) owns policy, selection, and fallback policy.
- **JS orchestration** (`src/**/*.js`) resolves/validates config, builds bind groups, creates pipelines, dispatches kernels, and handles readback.
- **WGSL execution** (`src/gpu/kernels/*.wgsl`) receives fully-resolved dispatch parameters and performs deterministic math only.

Any unresolved behavior choice is a contract error at runtime boundary, not an implicit runtime branch.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Orchestrator / Application (Policy Layer)                │
│  - Decides which models to query                                         │
│  - Chooses merge weights                                                 │
│  - Builds prompts (Seed/Reflect/Refine)                                  │
│  - Runs orchestration loops                                               │
│  - Calculates fitness scores                                             │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ Parameters (weights, prompts, config)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           DOPPLER (Engine)                               │
│  - Executes inference: prompt → tokens                                   │
│  - Manages GPU buffers and KV cache                                      │
│  - Merges logits on GPU (weighted, max, geometric)                       │
│  - Samples from distributions                                            │
│  - Returns results to caller                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multi-Model Primitives

When multiple models are loaded, these operations are exposed as DOPPLER primitives to avoid avoidable CPU round-trips:

| Operation | DOPPLER Primitive | Orchestrator Decides |
|-----------|-------------------|-----------------|
| Logit merging | `mergeMultipleLogits(logitBuffers, vocabSize, weights, temperature)` | Which buffers, what weights |
| KV cache sharing | `setSharedPrefix(prompt, options)` / `setSharedPrefixSnapshot(snapshot)` | When to share, which prefix |
| Sampling | `sample(logits, params)` (or GPU sample kernels when enabled) | Temperature, top-k, top-p |
| Expert execution | `executeExpert(expertId, prompt, options, overrides)` | Which expert, what prompt |
| Network execution | `executeGenome(genome, prompt, options, router)` | Genome structure |

### What Belongs Where

| Feature | DOPPLER (Engine) | Orchestrator (Policy) |
|---------|------------------|------------------|
| Inference | ✅ GPU kernels, buffer pools | ❌ |
| KV Cache | ✅ Memory management | ❌ |
| Logit Merge | ✅ GPU tensor ops | Weights, strategy |
| Prompt Templates | ❌ | ✅ "Review this code..." |
| Loop Logic | ❌ | ✅ for (i=0; i<turns; i++) |
| Expert Selection | ❌ | ✅ UCB1, bandit algorithms |
| Evolution | ❌ | ✅ GA, mutation, crossover |
| Fitness Scoring | ❌ | ✅ Quality heuristics |

### Migration Note

DOPPLER keeps orchestration separate from runtime inference. Use an external
policy layer (for example Reploid) with
`src/inference/multi-model-network.js` for cross-model coordination.

---

## Module Structure

Support tier is not implied by directory presence alone. Use the generated
[Subsystem Support Matrix](subsystem-support-matrix.md) for the public contract.

| Directory | Purpose |
|-----------|---------|
| `config/` | Schema, checked-in config assets, runtime config, manifest-first merge |
| `converter/` | GGUF/SafeTensors → RDRR conversion, quantization |
| `formats/` | GGUF, SafeTensors, RDRR parsing |
| `gpu/` | WebGPU device, buffer pool, WGSL kernels |
| `inference/` | Pipeline orchestration, KV cache, MoE routing |
| `loader/` | Weight loading, dequantization |
| `storage/` | OPFS shard management, download |
| `memory/` | Heap manager + Memory64/unified detection for loader/preflight |
| `client/` | Root facade and runtime coordination |
| `debug/` | Logging, trace categories, probes |
| `errors/` | Error codes and helpers |
| `rules/` | JSON rule maps for runtime selection |
| `experimental/` | Quarantined experimental and internal-only subsystem lanes such as adapters, orchestration, training, diffusion, bridge, browser helpers, hotswap, distribution, and energy |
| `types/` | Shared TypeScript types |

See private wrapper docs for optional wrapper-level architecture notes.

Direct-source artifact adaptation sits between `formats/` and the text runtime:
- `src/tooling/source-artifact-adapter.js` normalizes raw-source kind handling, dtype policy, and direct-source runtime-model construction.
- `src/storage/artifact-storage-context.js` owns file/HTTP artifact loading once a runtime model contract exists.
- Today the direct-source runtime path supports `safetensors`, `gguf`, `.tflite`, `.task`, and `.litertlm`.
- This direct-source path is part of the experimental subsystem support tier rather than the tier1 quickstart contract.
- Direct-source runtime planning uses source artifact facts as the storage truth. Converter-style quantization policy is only for actual RDRR conversion and is rejected on direct-source loads.
- `.tflite` support is currently a weight-container path over the existing transformer runtime: Doppler reads constant tensors directly from the FlatBuffer file, accepts both inline buffer vectors and LiteRT-style external buffer `offset/size` fields, requires sibling `config.json` metadata, and emits explicit `tensor.sourceTransform` metadata when LiteRT weights are per-tensor affine `INT8`, `UINT8`, or `INT4`.
- Some `.task` artifacts are raw TFLite FlatBuffers rather than ZIP bundles. Doppler detects that shape explicitly instead of assuming every `.task` file is a container archive.
- Those quantized constants stay compact on disk and are dequantized at load time to `F16`; per-channel, packed LiteRT-LM companion-tensor quantization (for example `*.w` plus `*.w_quantized_scale` / `*.sum_i`), or otherwise unsupported LiteRT quantization still fails closed.

---

## Architectural Views

DOPPLER's structure can be understood through multiple lenses. Each view serves a different purpose:

| View | Purpose | When to Use |
|------|---------|-------------|
| **Domain Grouping** | Mental model | Day-to-day orientation |
| **Dependency Graph** | True import relationships | Refactoring, circular dep analysis |
| **Pipeline View** | Runtime data flow | Debugging, performance tuning |
| **Client/API Layers** | Public entry to runtime orchestration | Onboarding, facade boundaries |

### Domain Grouping (Primary Mental Model)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               DOMAINS                                        │
├───────────────────┬────────────────────┬────────────────────────────────────┤
│      COMPUTE      │        DATA        │            RUNTIME                  │
├───────────────────┼────────────────────┼────────────────────────────────────┤
│ gpu/              │ formats/           │ inference/                          │
│ ├─ device         │ ├─ gguf            │ ├─ pipelines                        │
│ ├─ uniform-cache  │ ├─ safetensors     │ ├─ pipelines/text/attention          │
│ └─ kernels/       │ ├─ rdrr            │ ├─ pipelines/text/ffn                │
│    (WGSL kernels) │ └─ tokenizer       │ ├─ pipelines/text/logits             │
│                   │                    │ └─ kv-cache                         │
│                   │ loader/            │                                     │
│ memory/           │ ├─ doppler-loader  │ debug/                              │
│ ├─ buffer-pool    │ ├─ final-weights-loader │ ├─ log                          │
│ ├─ heap-manager   │ └─ shard-cache     │ └─ trace                            │
│ └─ capability     │                    │                                     │
│                   │                    │                                     │
│                   │ storage/           │                                     │
│                   │ ├─ shard-manager   │                                     │
│                   │ └─ quota           │                                     │
├───────────────────┴────────────────────┴────────────────────────────────────┤
│                          SHARED SERVICES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ config/                          │ gpu/kernels                               │
│ ├─ schema/ (all DEFAULT_*)       │ ├─ *.js (selection + dispatch)           │
│ ├─ conversion/                   │ ├─ kernel-configs.js (from registry)     │
│ ├─ runtime/                      │ └─ kernel-tuning.js (auto-benchmarking)  │
│ └─ runtime.js (get/set API)      │                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ types/                                                                       │
│ (shared TypeScript declarations - no runtime code)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      CLIENT + EXPERIMENTAL LANES                             │
├───────────────────┬────────────────────┬────────────────────────────────────┤
│ converter/        │ client/            │ experimental/                       │
│ (Node + Browser)  │ (Public API)       │ (quarantined advanced subsystems)   │
├───────────────────┼────────────────────┼────────────────────────────────────┤
│ experimental/     │ experimental/      │ experimental/                       │
│ adapters,hotswap  │ bridge,browser     │ diffusion,energy,training,dist.     │
└───────────────────┴────────────────────┴────────────────────────────────────┘
```

**Domain boundaries:**
- **Compute**: Stateless GPU transforms. Buffer management, raw kernels.
- **Data**: I/O, parsing, persistence. Format-aware, GPU-unaware.
- **Runtime**: Orchestration, state. Coordinates Compute and Data.
- **Shared Services**: Cross-cutting concerns used by multiple domains.
  - `config/`: Source of truth for all tunables (DEFAULT_* exports)
  - `gpu/kernels`: Selects kernel variants and dispatches work
  - `types/`: TypeScript declarations (compile-time only)
- **Client + experimental lanes**: `client/` is supported runtime coordination; quarantined advanced lanes live under `src/experimental/` and must be treated by the support-tier contract rather than by directory presence alone.

### Debug Infrastructure Layers

```
Browser harness (explicit suite context + runtime config)
  mode / workload / modelId
        |
Runtime Config (runtime.shared.tooling, runtime.shared.debug, runtime.shared.benchmark)
        |
  debug/            gpu/                   inference/pipelines/
  - log.js          - profiler.js           - kernel-trace.js
  - trace.js        - perf-guards.js
  - tensor.js
        |
Shared: debug/stats.js (median, p95, IQR, outliers)
```

### Dependency Graph (True Relationships)

This shows actual import dependencies. Use for refactoring decisions.

```
                              inference
                             /    |    \
                            /     |     \
                      loader    gpu      kv-cache
                        |      / | \        |
                     formats  /  |  \       |
                        |    /   |   \      |
                        ▼   ▼    ▼    ▼     ▼
                    ┌───────────────────────────┐
                    │      SHARED SERVICES      │
                    ├───────────────────────────┤
                    │  config/    kernels       │
                    │  schema     configs       │
                    │     │       tuning        │
                    └─────┼─────────────────────┘
                          │
                    ┌─────┼─────┐
                    ▼     ▼     ▼
                 types  memory  debug

    storage ◀─────────────────────────▶ (orthogonal to gpu)
```

**Key observations:**
- `inference` depends directly on `gpu` (skips `loader`). Intentional for kernel dispatch.
- `config/schema` is imported by almost everything. Source of truth for defaults.
- `kernel-configs` mediates between inference and raw WGSL kernels (derived from `src/config/kernels/registry.json`)
- `types` is compile-time only; `memory` can depend on `debug` for telemetry/logging.
- `storage` and `gpu` are orthogonal. Neither imports the other.

**Circular dependency risks:**
- `inference ↔ loader`: loader needs inference config, inference needs loaded weights
- Currently broken by: loader returns raw weights, inference builds pipeline config

### Pipeline View (Runtime Flow)

Use this view when debugging inference or optimizing performance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Tokenize │  Embed   │ Layer×N  │ LM Head  │  Sample  │      Decode         │
├──────────┴──────────┴────┬─────┴──────────┴──────────┴─────────────────────┤
│                          │                                                  │
│    ┌─────────────────────┴─────────────────────┐                           │
│    │              LAYER BLOCK                   │                           │
│    ├──────────┬──────────┬──────────┬──────────┤                           │
│    │ RMSNorm  │ Attn+KV  │ RMSNorm  │   FFN    │                           │
│    │          │  +RoPE   │          │ SiLU/GeGLU│                           │
│    └──────────┴──────────┴──────────┴──────────┘                           │
│                          │                                                  │
├──────────────────────────┼──────────────────────────────────────────────────┤
│       KERNEL LAYER       │              SUPPORT LAYER                       │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ matmul  │ rope  │ silu   │  config  │  debug  │  types  │  memory           │
│ attn    │ norm  │ gather │  loader  │ formats │ storage │                   │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

**Data flow per token:**
1. **Tokenize**: `src/inference/tokenizer.js` → token IDs
2. **Embed**: `src/gpu/kernels/gather.wgsl` → hidden state [seq, hidden_dim]
3. **Layer×N**: For each transformer layer:
   - RMSNorm → Attention (Q/K/V matmul, RoPE, softmax, output) → Residual
   - RMSNorm → FFN (gate, up, activation, down) → Residual
4. **LM Head**: `src/gpu/kernels/matmul_*` → logits [vocab_size]
5. **Sample**: GPU or CPU sampling path (execution-plan/runtime dependent) → next token ID
6. **Decode**: `src/inference/tokenizer.js` → output text

### Client/API Layering (Preferred View)

Use this to reason about the public facade, runtime orchestration, and domain subsystems without implying a pure foundation tier.

- `client/api`: the root `doppler` facade, convenience cache, load/chat/text helpers, and surface normalization.
- `client/runtime`: source resolution and materialization, device initialization, pipeline/session assembly, adapter lifecycle, and browser/Node adaptation.
- `domain subsystems`: `gpu`, `loader`, `storage`, `formats`, `memory`, `inference`, `rules`, `config`, `debug`, plus quarantined advanced lanes under `src/experimental/` such as `adapters`, `hotswap`, `bridge`, `browser`, `training`, `diffusion`, `distribution`, `orchestration`, and `energy`.

Layering is directional, not a promise of zero internal dependencies. Shared low-level helpers may depend on `config`, `debug`, or `gpu` when that is their owning contract.

**Testing implications:**
- `client/api` and most `client/runtime` flows can be exercised with mocked runtime inputs.
- `gpu` and `inference` integration paths require a WebGPU adapter.
- Full model-load flows require the relevant model artifacts and manifest/config inputs.

---


## 1. GPU Subsystem
Core GPU stack lives under `src/gpu/` and `src/memory/`. It covers device init,
buffer pooling, kernel selection, profiling, and scheduling.

## 2. Inference Pipeline (`inference/`)

### pipelines/text.js - Main Orchestration

The core generate loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Token Generation Loop                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Tokenize prompt                                              │
│  2. PREFILL: Process all prompt tokens in parallel               │
│     ├─ Embed tokens (gather)                                     │
│     ├─ For each layer:                                           │
│     │   ├─ RMSNorm (input)                                       │
│     │   ├─ QKV projections (matmul)                              │
│     │   ├─ RoPE (Q, K)                                           │
│     │   ├─ QK-Norm (Gemma 3)                                     │
│     │   ├─ Attention (fused)                                     │
│     │   ├─ O projection (matmul)                                 │
│     │   ├─ Residual add                                          │
│     │   ├─ RMSNorm (FFN)                                         │
│     │   ├─ FFN: gate_proj, up_proj, SiLU, down_proj              │
│     │   │   OR MoE: route → expert FFNs → combine                │
│     │   └─ Residual add                                          │
│     ├─ Final RMSNorm                                             │
│     ├─ LM head (matmul)                                          │
│     └─ Sample token                                              │
│  3. DECODE: Generate tokens one at a time                        │
│     ├─ Same flow but queryLen=1                                  │
│     ├─ KV cache stores previous K,V                              │
│     └─ Attention uses cached K,V                                 │
│  4. Yield token to caller                                        │
│  5. Check stop conditions                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Layer Pipeline Plans (experimental):**

The default layer order is fixed and optimized in
`src/inference/pipelines/text/layer.js`. For advanced experimentation, you can
supply a JSON plan under `runtime.inference.pipeline` (runtime override) or
`inference.pipeline` in the manifest. The plan executes via a small interpreter and
is slower than the default path.

Example (LLaMA-style residuals):

```json
{
  "inference": {
    "pipeline": {
      "steps": [
        { "op": "save", "name": "residual" },
        { "op": "attention" },
        { "op": "residual_add", "a": "state", "b": "residual" },
        { "op": "save", "name": "residual" },
        { "op": "rmsnorm", "weight": "post_attn" },
        { "op": "ffn" },
        { "op": "residual_add", "a": "state", "b": "residual" }
      ]
    }
  }
}
```

**Norm weight names** (canonical):
- `input` - input layernorm
- `post_attn` - post-attention norm (preferred). `post_attention` is kept only for legacy manifests.
- `pre_ffn` - pre-feedforward norm (Gemma 2/3 sandwich)
- `post_ffn` - post-feedforward norm (Gemma 2/3 sandwich)

**Skipping operations:**
- To skip input norm in attention: `{ "op": "attention", "skipInputNorm": true }`
- To skip FFN entirely: omit the `ffn` step, or use `{ "op": "noop" }` as placeholder
- Per-layer overrides can provide different step lists for specific layers

Example (skip FFN on layer 0):
```json
{
  "steps": [
    { "op": "save", "name": "residual" },
    { "op": "attention" },
    { "op": "residual_add", "a": "state", "b": "residual" }
  ],
  "overrides": [
    { "layers": [0], "steps": [
      { "op": "attention" }
    ]}
  ]
}
```

**Runtime profiles:** See `src/config/runtime/profiles/` for complete runtime overlay examples.

### multi-pipeline-pool.js - Concurrent Pipelines

Manages a pool of inference pipelines to run multiple requests in parallel.

### multi-model-network.js - Expert Network Orchestration

Schedules a network of experts and routes tasks across multiple models.

### expert-router.js - Expert Profiles

Defines expert profiles and routing hints used by multi-model orchestration.

### network-evolution.js - Topology Search

Evolution helpers for mutating and scoring expert network topologies.

**Key Pipeline Features:**
- GPU-native hot path: most tensor ops remain on GPU; logits readback cadence is controlled by runtime batching config
- GQA support: Multiple Q heads share K,V heads
- Sandwich norms: Gemma 3 pre/post FFN norms
- YARN RoPE: Extended context via per-dimension scaling

### Pipeline Submodules (`inference/pipelines/`)

| Module | Purpose |
|--------|---------|
| `src/inference/pipelines/registry.js` | Pipeline registry and factory |
| `src/inference/pipelines/context.js` | Runtime context and mode settings |
| `src/inference/pipelines/text/config.js` | Pipeline configuration and model params |
| `src/inference/pipelines/text/state.js` | Runtime state container |
| `src/inference/pipelines/text/generator.js` | Main generation loop |
| `src/inference/pipelines/text/generator-steps.js` | Step orchestration for prefill and decode |
| `src/inference/pipelines/text/generator-helpers.js` | Shared generation helpers |
| `src/inference/pipelines/text/layer.js` | Per-layer processing |
| `src/inference/pipelines/text/embed.js` | Token embedding |
| `src/inference/pipelines/text/sampling.js` | Token sampling (top-k, top-p, temperature) |
| `src/inference/pipelines/text/kernel-trace.js` | Kernel trace and instrumentation |
| `src/inference/pipelines/text/probes.js` | Debug probes |
| `src/inference/pipelines/text/attention/` | Attention kernels and helpers |
| `src/inference/pipelines/text/ffn/` | Feed-forward kernels and helpers |
| `src/inference/pipelines/text/logits/` | Logits projection |

**Note:** `src/inference/pipelines/text.js` composes these modules. Some helpers are used indirectly via `src/inference/pipelines/text/init.js` and `src/inference/pipelines/text/generator-steps.js`.

### Manifest-First Config Architecture

DOPPLER uses a **manifest-first** architecture where all model-specific inference parameters are embedded in the manifest at conversion time. This makes the manifest the single source of truth for how to run inference on a model.

**Data Flow:**

```
CONVERSION TIME:
┌─────────────────────────────────────────────────────────────────────┐
│ HuggingFace/GGUF Model → tooling convert surface                    │
│   - Node CLI: node-command-runner.js → node-converter.js            │
│   - Browser: browser-command-runner.js + convertHandler             │
│   - Build ManifestInferenceSchema from explicit conversion config + HF config │
│   - Write inference config to manifest.json                         │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                        manifest.json
                    (inference field required)

LOAD TIME:
┌─────────────────────────────────────────────────────────────────────┐
│ manifest.json + runtime overrides → mergeConfig()                   │
│   - Manifest inference values are required (source of truth)        │
│   - Runtime can override any manifest inference value               │
│   - Source tracking: '_sources' map shows where each value came from│
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                         MergedConfig
                   (all values resolved with sources)

EXECUTION TIME:
┌─────────────────────────────────────────────────────────────────────┐
│ Pipeline reads from ParsedModelConfig                               │
│   - No model detection (isGemma, isLlama, etc.)                     │
│   - Config trace shows source of every inference value              │
└─────────────────────────────────────────────────────────────────────┘
```

**Manifest Inference Fields:**

| Section | Fields | Example (Gemma 2) |
|---------|--------|-------------------|
| `attention` | `queryPreAttnScalar`, `attnLogitSoftcapping`, `slidingWindow`, `queryKeyNorm` | `256`, `50`, `4096`, `false` |
| `normalization` | `rmsNormWeightOffset`, `postAttentionNorm`, `preFeedforwardNorm`, `postFeedforwardNorm` | `true`, `true`, `true`, `false` |
| `ffn` | `activation`, `gatedActivation` | `"gelu"`, `true` |
| `rope` | `ropeTheta`, `ropeLocalTheta`, `ropeScalingType`, `ropeScalingFactor` | `10000`, `null`, `null`, `1.0` |
| `output` | `finalLogitSoftcapping`, `tieWordEmbeddings`, `scaleEmbeddings` | `30`, `false`, `true` |
| `layerPattern` | `type`, `globalPattern`, `period` | `"alternating"`, `"odd"`, `null` |

**Layer Pattern Transformation (Conversion Config → Manifest):**

Conversion configs define layer patterns in one format; the converter transforms them for the manifest:

| Conversion Config Format | Manifest Format | Example |
|---------------|-----------------|---------|
| `type: "all_attention"` | `type: "uniform"` | All layers use global attention |
| `type: "alternating", globalPattern: "odd"` | `type: "alternating"` | Odd layers use global attention |
| `type: "alternating", globalPatternN: 6` | `type: "every_n", period: 6` | Every 6th layer uses global attention (Gemma 3) |

The converter (`src/converter/manifest-inference.js`) performs this mapping. Conversion configs use `globalPatternN` for readability; manifests use `period` for runtime efficiency.

**RoPE Theta Sourcing:**

`ropeTheta` has a defined precedence during conversion (`src/converter/rope-config.js`):

1. **HuggingFace config** (`config.rope_theta`). Source of truth.
2. **Conversion config** (`converterConfig.inference.rope.ropeTheta`). Fallback.
3. **Default** (`10000`). Last resort.

This means Gemma 2 models don't hardcode `ropeTheta` in a separate family registry; the value comes from the HuggingFace config during conversion. Gemma 3 conversion configs may set `ropeTheta: 1000000` explicitly because it's a defining characteristic.

`ropeLocalTheta` precedence is:
1. `config.rope_parameters.sliding_attention.rope_theta` (when present)
2. conversion config `inference.rope.ropeLocalTheta`
3. `null`
Source: `src/converter/rope-config.js`.

**Source Tracking (`ConfigSource`):**
- `'manifest'` - Value came from manifest (converter output)
- `'runtime'` - Value was overridden by user at runtime

**Nullable Required Fields:**
- `null` = explicitly disabled (valid)
- `undefined` = not specified (validation error)
Runtime overrides apply whenever a runtime value is present (`!== undefined`),
including explicit `null` values. This behavior is implemented in `src/config/merge.js`.
For text-style manifests, `manifest.architecture` presence is required.
Field-level architecture checks run when `architecture` is present as an object
in `src/formats/rdrr/validation.js`.

**Benefits:**
- Manifest is self-describing: no need for external family-registry files
- Config tracing shows exactly where each value came from
- Model detection happens once at conversion, not at runtime
- Runtime overrides are explicit and traceable

**Migration:**
Models converted before the manifest-first architecture will fail validation with a helpful error:
```
Manifest for "model-name" is missing required 'inference' field.
This model was converted with an older version of DOPPLER.
Please re-convert the model using the latest converter.
```

### kv-cache.js - KV Cache Management

```javascript
// Cache structure per layer
{
  k: GPUBuffer,  // [maxSeqLen, numKVHeads, headDim]
  v: GPUBuffer,  // [maxSeqLen, numKVHeads, headDim]
  seqLen: number // Current filled length
}
```

**Layouts:**
- `contiguous`: Single buffer per K/V (default for <8K)
- `paged`: Block-based allocation for long contexts
- `tiered`: Hot ring + cold pages (optional tiered storage tracking)

Paged and tiered layouts keep K/V in GPU buffers; sliding windows or RoPE scaling still cap active context.

**KV dtype:** F16 when supported, halves VRAM usage.

### tokenizer.js - Tokenization

Loads tokenizer from:
1. Bundled tokenizer.json in model directory
2. HuggingFace-format vocab files

Supports chat templates via tokenizer_config.json.

### moe-router.js - Mixture of Experts

```
┌─────────────────────────────────────────────────────────────────┐
│                     MoE Layer Flow                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Router: hidden → [batch, num_experts] logits                 │
│  2. Softmax + Top-K: Select top-2 experts per token              │
│  3. Gather: Route tokens to their experts                        │
│  4. Expert FFN: Each expert processes its tokens                 │
│  5. Scatter-add: Combine weighted expert outputs                 │
└─────────────────────────────────────────────────────────────────┘
```

GPU-native routing avoids CPU readback of routing decisions.

---

## 3. Loader (`src/loader/doppler-loader.js`)

### Weight Loading Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   Weight Loading Flow                            │
├─────────────────────────────────────────────────────────────────┤
│  Shard (OPFS)                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────┐                                        │
│  │ Load raw bytes      │                                        │
│  │ (Q4_K_M / BF16)     │                                        │
│  └─────────────────────┘                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────┐                                        │
│  │ Upload to GPU       │  (staging buffer)                      │
│  │ as quant buffer     │                                        │
│  └─────────────────────┘                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────┐                                        │
│  │ Dequantize on GPU   │  (dequant_shared.wgsl)                 │
│  │ Q4_K → F32/F16      │                                        │
│  └─────────────────────┘                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────┐                                        │
│  │ Downcast to F16     │  (if hasF16, for matmul weights)       │
│  │ (optional)          │                                        │
│  └─────────────────────┘                                        │
│       │                                                          │
│       ▼                                                          │
│  GPU Buffer ready for inference                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-shard tensors:** Large tensors span multiple 64MB shards. Loader streams spans directly to GPU to avoid JS heap exhaustion.

**Gemma 2/3 norm offset:** RMSNorm uses `(1 + weight) * x` instead of `weight * x`. This is applied **at runtime** via the `rmsNormWeightOffset` flag passed to RMSNorm kernels. It is not applied during weight loading. The manifest `inference.normalization.rmsNormWeightOffset` field controls this behavior.

### multi-model-loader.js - Base + Adapter Loading

Loads a base model plus adapters (LoRA) for multi-model and expert scenarios.

---

## 4. Storage (`storage/`)

### RDRR Format (`formats/rdrr/`)

Custom model format optimized for browser streaming:

Note: RDRR helpers are canonically exported from `src/formats/rdrr/index.js`.
`src/storage/shard-manager.js` keeps a limited compatibility re-export (`getManifest`) for existing callers.
Shared file/HTTP artifact loading for both `rdrr` and persisted direct-source artifacts lives in
`src/storage/artifact-storage-context.js`; direct-source bundles enter the text runtime as
runtime-model contracts rather than synthetic RDRR manifests.

```
model-directory/
├── manifest.json          # Tensor locations, config, hashes
├── shard_00000.bin        # 64MB shard
├── shard_00001.bin
├── ...
└── tokenizer.json         # Optional bundled tokenizer
```

**manifest.json structure (illustrative subset):**
Use `docs/rdrr-format.md` and `src/formats/rdrr/validation.js` as canonical contract sources.
```json
{
  "modelId": "gemma-3-1b-q4",
  "version": "1.0",
  "quantization": "q4k",
  "totalSize": 1073741824,
  "hashAlgorithm": "blake3",
  "config": { /* HuggingFace config */ },
  "inference": { /* required model-specific inference config */ },
  "tensors": {
    "model.embed_tokens.weight": {
      "shard": 0,
      "offset": 0,
      "size": 8388608,
      "shape": [262144, 1536],
      "dtype": "Q4_K_M"
    }
  },
  "shards": [
    { "filename": "shard_00000.bin", "size": 67108864, "hash": "..." }
  ]
}
```

### shard-manager.js - Storage Backends

Uses config-driven storage backends with OPFS as the preferred path and IndexedDB/memory transport backends selected by explicit storage policy:
- `initStorage()` - Initialize selected backend
- `openModelStore(modelId)` - Set active model storage
- `loadShard(idx)` - Read shard to ArrayBuffer
- `loadShardRange(idx, offset, length)` - Range read (avoid materializing whole shard)
- `streamShardRange(idx, offset, length, { chunkBytes })` - Chunked range stream
- `verifyIntegrity()` - Check all shard hashes
- `computeHash(data, algo)` - Blake3/SHA256
- `listFilesInStore()` - List files in the currently-open model directory
- `loadFileFromStore(filename)` - Read an arbitrary file from the open model
- `streamFileFromStore(filename, { chunkBytes })` - Stream a file in chunks (no full-file RAM load)

### export.js - Export From Local Storage

Exports a stored model (manifest + shards + tokenizer artifacts) from OPFS/IndexedDB to a user-chosen directory using the File System Access API:
- `exportModelToDirectory(modelId, FileSystemDirectoryHandle, { chunkBytes, onProgress })`

This is designed to be safe for large models: the export path streams bytes and does not require loading entire shards into memory.

### downloader.js - Model Download

Streaming download with:
- Progress callbacks
- Shard-by-shard integrity verification
- Resume support (partial downloads)

### quickstart-downloader.js - Curated Downloads

Provides a curated model list and helpers for quickstart downloads.

### preflight.js - Requirements Check

Validates GPU and storage requirements before loading a model.

### quota.js - Storage Detection

Detects available storage APIs and reports quota/persistence information.

---

## 5. Memory Subsystem (`memory/`)

Memory capabilities inform loader/preflight/heap strategy (not kernel selection). The loader
and client API use these signals to size host memory allocations and report limits.

### capability.js - Memory Detection

Detects runtime capabilities:
```javascript
{
  hasMemory64: boolean,        // WebAssembly Memory64
  isUnifiedMemory: boolean,    // Unified memory GPUs (Apple/AMD)
  unifiedMemoryInfo: object,   // Unified memory metadata
  maxHeapSize: number | null,  // Max heap size (Memory64 only)
  segmentedLimits: object | null, // Segment caps (non-Memory64)
  strategy: 'MEMORY64' | 'SEGMENTED',
}
```

### unified-detect.js - Unified Memory Detection

Apple Silicon detection for optimal buffer sharing:
- Unified memory allows larger models (no PCIe copy)
- Detected via `navigator.gpu.requestAdapter()` heuristics

### heap-manager.js - Heap Allocation

Manages host-memory allocations for weight staging and conversion:
- Memory64: single large WASM heap
- Segmented: multiple ArrayBuffers with virtual addressing

---

## 6. Bridge (`bridge/`)

### extension-client.js - Extension Bridge Client

Connects to a browser extension to access local files outside OPFS limits.

### extension/background.js - Extension Background

Handles native messaging and file operations for the extension bridge.

### extension/manifest.json - Extension Manifest

Defines extension permissions and background entry points.

### Native Host (removed)

The native host process and install scripts were removed. Current workflows use
the extension bridge for browser contexts and the Node CLI for scripted/local
tooling paths.

### protocol.js - Bridge Protocol

Defines framing, commands, and flags for bridge messaging.

---

## 7. Browser Import (`browser/`)

### file-picker.js - Browser File Access

Uses browser file system APIs to pick and stream local model files.

### gguf-parser-browser.js - GGUF Parsing

Parses GGUF metadata and tensors in the browser.

### safetensors-parser-browser.js - SafeTensors Parsing

Parses SafeTensors metadata and tensor slices in the browser.

### gguf-importer.js - GGUF Import Pipeline

Builds RDRR shards from GGUF inputs in-browser.

### browser-converter.js - Browser Conversion

Converts source formats into RDRR with progress reporting. Uses shared types and functions from `src/converter/core.js` for consistent manifest generation and architecture extraction.

---

## 8. Converter (`converter/`)

### core.js - Shared Conversion Core

Platform-agnostic types and pure functions shared across converter paths:
- **Types**: `TensorInfo`, `ParsedModel`, `ConvertOptions`, `RDRRManifest`, `ShardInfo`, `TensorLocation`
- **Functions**: `sanitizeModelId()`, `formatBytes()`, `shouldQuantize()`, `extractArchitecture()`, `buildTensorMap()`, `createManifest()`
- **I/O Adapter**: `ConvertIO` interface for platform-specific file operations

### browser-converter.js - Model Conversion

Converts HuggingFace models to RDRR format in-browser (demo UI or programmatic API).

### quantizer.js - Q4_K Quantization

**Critical:** Must match llama.cpp Q4_K format exactly.

```javascript
// llama.cpp dequantization formula:
value = d * scale * q - dmin * min

// Where:
// - d, dmin: per-block scale factors (f16)
// - scale, min: per-subblock (6-bit packed)
// - q: 4-bit quantized value (0-15)
// - min is stored as positive offset to subtract
```

**Post-mortem note:** Early bug stored `min` with different sign convention,
causing all dequantized values to be positive. Internal postmortem notes track
the full debugging timeline.

---

## 9. Tools

DOPPLER ships Node tooling via `src/cli/doppler-cli.js` and npm scripts
(`convert`, `debug`, `bench`, `verify:model`). The demo UI and browser harness
remain the primary interactive diagnostics surfaces.

---

## 10. Client API (`client/api`)

Primary application-facing API for model loading and generation via the root facade:

```javascript
import { doppler } from 'doppler-gpu';

const model = await doppler.load('gemma3-270m');

for await (const token of model.generate('Hello')) {
  console.log(token);
}

const response = await model.chatText([
  { role: 'user', content: 'Write one sentence about WebGPU.' },
]);
```

This layer owns convenience caching, facade normalization, and public call-shape validation. It does not expose the legacy singleton surface as the primary entry point.

For runtime capability classes and sizing guidance, use the canonical doc:
`docs/performance-sizing.md`.

---

## Data Flow: Single Token Generation

For detailed kernel-level execution trace including tensor shapes, kernel selection, and fusion analysis, see `../src/inference/README.md`.
The walkthrough below is an illustrative text-model example (Gemma-like dimensions), not a universal shape for all model families.

```
User prompt: "Hello"
         │
         ▼
┌────────────────────┐
│ Tokenizer.encode() │ → [1, 15043]  (BOS + "Hello")
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ GPU: gather        │ → embeddings[2, 1536]
│ (embed_tokens)     │
└────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│ For each of 26 layers:                             │
│   ┌──────────────┐                                 │
│   │ RMSNorm      │ hidden[2, 1536]                │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ Q projection │ matmul → Q[2, 4, 256]          │
│   │ K projection │ matmul → K[2, 1, 256]  (GQA)   │
│   │ V projection │ matmul → V[2, 1, 256]          │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ RoPE         │ Apply positional encoding       │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ Attention    │ Q@K^T → softmax → @V           │
│   │ (fused)      │ Output: [2, 4, 256]            │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ O projection │ matmul → [2, 1536]             │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ Residual add │ hidden += attn_out             │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ RMSNorm      │ (pre-FFN)                      │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ FFN          │ gate*up → SiLU → down          │
│   │ (SwiGLU)     │ [2, 6144] → [2, 1536]          │
│   └──────────────┘                                 │
│          │                                         │
│          ▼                                         │
│   ┌──────────────┐                                 │
│   │ Residual add │ hidden += ffn_out              │
│   └──────────────┘                                 │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│ Final RMSNorm      │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ LM Head matmul     │ → logits[262144]
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ CPU: Sample        │ → token_id = 1247 ("world")
│ (top-k, top-p)     │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Tokenizer.decode() │ → "world"
└────────────────────┘
```

---

## Key Design Decisions

### 1. GPU-Native Pipeline
Hot-path tensor operations execute on GPU, with readback cadence controlled by runtime batching policy. This minimizes CPU↔GPU transfers during decode.  
Source: `src/inference/pipelines/text/generator-steps.js`, `src/inference/pipelines/text/attention/run.js`

### 2. Q4_K Quantization
4-bit quantization significantly reduces storage footprint relative to f16/f32 weights, with quality/perf tradeoffs depending on workload and model.  
Source: `src/converter/quantizer.js`, `src/formats/rdrr/validation.js`

### 3. 64MB Shards
Shard size balances:
- Small enough for reliable streaming download
- Large enough to minimize request overhead
- Aligned with OPFS block allocation
  
Source: `src/config/schema/manifest.schema.js` (`SHARD_SIZE`)

### 4. Streaming Weight Load
Large tensors (embeddings, LM head) are streamed directly to GPU buffers to avoid JS heap exhaustion.
Source: `src/loader/doppler-loader.js`, `src/loader/tensors/tensor-loader.js`

### 5. Capability-Based Kernel Selection
Different devices get different kernel implementations:
- F16 hardware → F16-capable kernel variants
- Subgroup support → shuffle-based reductions
- Large context → streaming attention

Source: `src/rules/inference/kernel-path.rules.json`, `src/rules/kernels/*.rules.json`, `src/gpu/kernels/attention.js`  
See `config.md` for kernel selection rules and runtime overrides.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/client/doppler-api.js` | Root facade implementation |
| `src/client/doppler-api.d.ts` | Root facade types |
| `src/inference/pipelines/text.js` | Main inference orchestration |
| `src/inference/pipelines/text/generator.js` | Generation loop |
| `src/inference/kv-cache/index.js` | KV cache management |
| `src/inference/tokenizer.js` | Tokenizer selection and initialization |
| `src/inference/moe-router.js` | MoE expert routing |
| `src/loader/doppler-loader.js` | Weight loading and dequant |
| `src/gpu/device.js` | WebGPU initialization |
| `src/gpu/kernels/index.js` | Kernel selection and dispatch exports |
| `src/gpu/kernel-tuner.js` | Kernel tuning harness |
| `src/memory/buffer-pool.js` | Buffer pooling |
| `src/formats/rdrr/manifest.js` | RDRR manifest parsing |
| `src/storage/shard-manager.js` | OPFS shard management |
| `src/converter/core.js` | Shared conversion types and functions |
| `src/converter/quantizer.js` | Q4_K quantization |
| `src/experimental/browser/browser-converter.js` | Browser model conversion |

---

## Related Documentation

- `rdrr-format.md` - RDRR format specification

---

## Tokenizer Boundary Notes

- `src/formats/tokenizer/` owns tokenizer file parsing and metadata normalization.
- `src/inference/tokenizer.js` owns tokenizer runtime setup and output shaping.
- `src/inference/pipelines/text/chat-format.js` owns prompt-format-specific behavior.
- Keep format parsing and runtime inference logic in separate layers; avoid importing
  tokenizer parser internals directly into inference execution paths.

*Last reviewed for repo accuracy: 2026-03-05*

## Kernel Overrides & Compatibility
See `operations.md#kernel-overrides--compatibility` (canonical section).



## Execution Pipeline & System Flow
Detailed token-level execution walkthroughs are maintained in wrapper notes at
private wrapper docs.
