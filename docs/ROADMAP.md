# DOPPLER Roadmap

## Vision

**Why DOPPLER exists:** Browser-native, dynamic LLM inference that can't be achieved with pre-compiled approaches like TVM/WebLLM.

---

## Current Priorities (Q1 2026)

Execution sequence:

1. **P0:** Gemma-3 inference correctness smoke test (prompt + pass/fail).
2. **P1:** M1 latency benchmark for Gemma-3-1B Q4K (publish hardware + config).
3. **P2:** IntentBundle parity gate (Dynamic Mode evidence pipeline).
4. **P3:** Static Mode enforcement (locked kernels, signed bundles, audit output).

These stay in Doppler; Reploid tracks the SAB loop, HITL wiring, and VFS proof emission.

---

## The Capability Thesis

The industry is fixated on a false dichotomy: **privacy vs. performance**. That is boring.

Reploid's value proposition is **Capability**. The architecture is designed to eventually run a 600B+ parameter model on a MacBook Air without lobotomizing the weights.

We achieve this by inverting the standard stack. Instead of bringing the data to a centralized model, we mount a **Distributed Mixture-of-Experts (MoE)** directly to the browser runtime via DOPPLER. The P2P mesh becomes an infinite-capacity cache tier for model weights.

**Vision (Phase 3+):**

- **Infrastructure:** A 600B+ parameter MoE mounted over P2P WebRTC mesh
- **Intelligence:** Hierarchical routers that stream specialized expert clusters on demand
- **Evolution:** LoRA adapters distributed as delta shards that upgrade model capabilities

> **Current state:** Phase 1 (performance parity with 1-3B models) is in progress. The 600B+ MoE capability is Phase 3+ — see roadmap below.

We trade bandwidth (which is cheap) for intelligence (which is expensive). This delivers datacenter-grade capability on consumer-grade hardware. The big players can't pivot because their valuation depends on renting H100s.

---

## The Scale Math

A frontier-class model (e.g., DeepSeek-V3, 671B parameters) is not a monolithic binary. It is a file system of thousands of granular expert shards.

**RDRR Sharding Strategy:**
- Model sliced into ~9,600 **Expert Shards** (64MB each)
- 64MB optimized for WebRTC `RTCDataChannel` throughput
- Aligns with browser OPFS block allocation

**Content Addressing:**
- Request `SHA256(shard_bytes)`, not "Health Expert v1"
- Instant integrity verification
- P2P mesh acts as infinite-capacity L4 cache

**The Mount:**
- Reploid downloads a **Manifest** (~150KB), not the model
- Weights stay on the network until called
- MoE sparsity: only ~25% of experts active per token

**Tiered Storage:**

| Tier | Capacity | Latency | Contents |
|------|----------|---------|----------|
| GPU VRAM | 8-24GB | <1ms | Active experts, KV cache |
| Unified RAM | 32-128GB | ~5ms | Warm experts, session state |
| OPFS | 10-50GB | ~50ms | Cold experts, cached shards |
| P2P Swarm | Unlimited | ~200ms | Rare experts, full model |

See `ARCHITECTURE.md` for tiered memory and cache layout notes.

**Result:** 600B model on consumer hardware.

**MoE Inference Research:**
- [MoBiLE (arXiv:2510.12357)](https://arxiv.org/abs/2510.12357): 1.6-1.7× speedup via mixture of big-little experts on consumer GPU
- [HybriMoE (DAC 2025)](https://arxiv.org/abs/2504.05897): Hybrid CPU-GPU scheduling with 1.33× prefill / 1.70× decode speedup

---

## Architectural Bets

DOPPLER makes five interlocking bets that diverge from the WebLLM/TVM approach:

### Bet 1: Code/Data Separation

| Layer | WebLLM | DOPPLER |
|-------|--------|---------|
| GPU kernels | WGSL (TVM-generated, model-specific) | WGSL (generic, shared across models) |
| Orchestration | WASM (compiled C++) | JavaScript |
| Weights | Bundled with compiled binary | Just data (64MB shards) |

WebLLM compiles each model into a `.wasm` bundle containing TVM-generated WGSL shaders + WASM orchestration code. The model is a **monolithic unit**.

DOPPLER treats kernels as **generic data processors**. The same `matmul.wgsl` processes any weight matrix. Models are just **manifest + weight shards**.

> "MLC-LLM takes in any open-source model's implementation in Python... and compiles the model's computation into the backend of interest (WebGPU). The WASM library contains both compute kernels in WGSL and non-kernel functions in WebAssembly."
>
> Source: [MLC Blog](https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine)

### Bet 2: GPU Fusion Eliminates CPU Bottleneck

If 99% of compute is on GPU, optimizing the 1% CPU layer is pointless.

```
WebLLM decode step:        DOPPLER decode step:
├─ WASM tokenize           ├─ JS: bind buffers
├─ WASM layer loop         ├─ JS: queue.submit()
├─ WASM attention logic    ├─ GPU: fused attention
├─ WASM sampling           ├─ GPU: fused MoE routing
└─ ~2-3x faster CPU ops    ├─ JS: read logits (0.5ms)
                           └─ JS: sample token

CPU time: significant      CPU time: ~0.5ms (negligible)
```

**DOPPLER's MoE routing is GPU-native**: Router → softmax+topk → scatter_add all stay on GPU with zero CPU readback. See [gpu/kernels/topk.wgsl](../src/gpu/kernels/topk.wgsl).

### Bet 3: JavaScript is Fast Enough

Modern JS engines (V8, SpiderMonkey, JSC) are extraordinarily optimized:
- JIT compilation approaches native speed for hot paths
- TypedArrays enable zero-copy GPU upload
- Async/await enables non-blocking GPU dispatch

For thin orchestration (dispatch commands, sample one token), JS overhead is ~0.5ms per decode step. This is **2%** of a 26ms decode cycle.

> "Chrome's V8 JavaScript engine and Firefox's SpiderMonkey... achieve performance within 50-80% of native code for compute-intensive workloads."
>
> Source: [WebAssembly vs JavaScript Performance (Chrome)](https://chromium.googlesource.com/chromium/src/+/refs/heads/main/third_party/WebAssembly/README.md)

### Bet 4: Readback is the Real Enemy

Every GPU→CPU transfer costs 2-6ms (wait + copy + map), 10-60x the actual compute time.

| Operation | DOPPLER | Readback? |
|-----------|---------|-----------|
| Attention scores | Fused in `attention.wgsl` | No |
| MoE routing | Fused in `softmax_topk.wgsl` | No |
| KV cache update | GPU buffer writes | No |
| Intermediate activations | Stay on GPU | No |
| Final logits | Read for sampling | **Yes (unavoidable)** |

DOPPLER's architecture ensures only ONE readback per token: the final logits for sampling.

### Bet 5: Flexibility Enables Impossible Capabilities

| Capability | Requires | WebLLM | DOPPLER |
|------------|----------|--------|---------|
| 90GB MoE on 8GB VRAM | Expert paging | **No** | **Yes** |
| P2P kernel evolution | Plain-text WGSL | **No** | **Yes** |
| LoRA without recompile | Generic kernels | **No** | **Yes** |
| Device-specific optimization | Runtime kernel swap | **No** | **Yes** |
| Speculative decoding | Multi-model coordination | Awkward | **Native** |

---

## Why Not TVM? (Detailed)

WebLLM uses Apache TVM to pre-compile model-specific WGSL kernels with auto-tuning:

| Aspect | TVM/WebLLM | DOPPLER |
|--------|------------|---------|
| Kernel performance | ~80% native (auto-tuned) | ~60-70% native (manual) |
| New model support | Requires offline compilation | Runtime-compatible |
| Dynamic sharding | Not possible (fixed in binary) | Load/unload experts freely |
| P2P distribution | Distribute compiled .wasm | Distribute weight shards only |
| Model evolution | Recompile entire model | Swap shards, keep kernels |
| LoRA adapters | Recompile with LoRA fused | Hot-swap at runtime |
| Browser-only operation | Needs compilation toolchain | Fully in-browser |

> "Evaluations show that WebLLM can retain up to 80% native performance on the same device."
>
> Source: [WebLLM: A High-Performance In-Browser LLM Inference Engine (arXiv:2412.15803)](https://arxiv.org/abs/2412.15803)

**Additional WebGPU Inference Research:**
- [WeInfer (WWW 2025)](https://dl.acm.org/doi/10.1145/3696410.3714553): 3.76× speedup over WebLLM via buffer reuse and async pipelining
- [nnJIT (MobiSys 2024)](https://dl.acm.org/doi/10.1145/3643832.3661892): JIT kernel generation for in-browser inference, 8.2× faster than baselines

**DOPPLER accepts ~20% kernel performance gap** because:
1. Large MoE models can't fit in VRAM anyway (paging required)
2. P2P distribution requires code/data separation
3. Runtime flexibility enables speculative decoding, LoRA, expert paging
4. These capabilities are **impossible** with pre-compiled approaches

---

## Concrete Capability Examples

### Expert Paging (MoE)

Mixtral 8x7B: 8 experts × ~3GB each = 24GB. Only 2 active per token.

```
WebLLM: All 8 experts compiled into binary
        └─ Must fit all 24GB in VRAM or fail

DOPPLER: Generic FFN kernel + weight buffer binding
        ├─ Load expert_2 from OPFS → bind → run kernel
        ├─ Evict expert_2 (release buffer)
        ├─ Load expert_7 from peer → bind → same kernel
        └─ Run 90GB model on 8GB VRAM via paging
```

### P2P Evolution (Dynamic Components)

The real P2P value is **evolving dynamic components**, not distributing static weights:

```
WebLLM: Kernels frozen in compiled binary
        └─ Everyone uses same TVM-generated kernels forever

DOPPLER: Kernels are plain text, evolve across swarm
        ├─ User discovers 2x faster attention on M3 Max
        ├─ Shares kernel hash → peers benchmark → confirm
        ├─ Best kernels propagate by device class
        └─ Also: LoRA adapters, sampling strategies, router weights
```

**Static weights** → CDN (HuggingFace). **Dynamic components** → P2P evolution.

See `OPERATIONS.md` for current benchmarks and constraints.

### Speculative Decoding

```
WebLLM: Draft model = separate compiled binary
        Target model = separate compiled binary
        └─ Coordinating two compiled models is awkward

DOPPLER: Draft model = small model weights
        Target model = large model weights
        ├─ Same kernels for both
        ├─ Share KV cache buffers
        └─ JS coordinates verification (flexible, debuggable)
```

---

## Phased Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1** | Performance Parity | In Progress |
| **2** | MoE Efficiency | Partial |
| **3** | Scale Beyond WebLLM | Planned |
| **4** | P2P Distribution | Design |
| **5** | Evolution | Design |

**Task tracking:** See `feature-log/doppler/*.jsonl` for detailed task database.

```
Phase 1: Performance Parity ──┐
                              ├──▶ Phase 3: Scale Beyond WebLLM
Phase 2: MoE Efficiency ──────┤
                              ├──▶ Phase 4: P2P Distribution
                              │
                              └──▶ Phase 5: Evolution
```

---

## Success Criteria

| Phase | Criteria | Validation |
|-------|----------|------------|
| **1** | 40+ tok/s on Gemma 3 1B | Benchmark harness |
| **2** | Mixtral 8x7B with expert paging | E2E test |
| **3** | 40GB+ model on 16GB unified mem | Memory profiling |
| **4** | 10-peer swarm self-heals | P2P integration test |
| **5** | LoRA personalization working | User preference test |

---

## Related Documents

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current system design |
| [CONFIG.md](CONFIG.md) | Kernel paths and runtime config |
| [OPERATIONS.md](OPERATIONS.md) | Troubleshooting and perf investigations |
| [POSTMORTEMS.md](POSTMORTEMS.md) | Incident summaries |

---

## References

### WebGPU & Browser Inference
1. MLC Team. (2024). [WebLLM: A High-Performance In-Browser LLM Inference Engine](https://arxiv.org/abs/2412.15803). arXiv.
2. Chen, Z. et al. (2025). [WeInfer: Unleashing the Power of WebGPU on LLM Inference](https://dl.acm.org/doi/10.1145/3696410.3714553). ACM WWW 2025.
3. Jiang, S. et al. (2024). [nnJIT: Empowering In-Browser Deep Learning Inference](https://dl.acm.org/doi/10.1145/3643832.3661892). MobiSys 2024.
4. Google. (2024). [WebAssembly and WebGPU Enhancements for Web AI](https://developer.chrome.com/blog/io24-webassembly-webgpu-1). Chrome Developers.

### MoE & Expert Paging
5. Zhao, Y. et al. (2025). [MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU](https://arxiv.org/abs/2510.12357). arXiv.
6. Zhang, Y. et al. (2025). [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management](https://arxiv.org/abs/2504.05897). DAC 2025.
7. NVIDIA. (2025). [Mixture of Experts Powers Frontier AI Models](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/).

### Browser Capabilities
8. [WebGPU Browser Support (Can I Use)](https://caniuse.com/webgpu)
9. [SharedArrayBuffer (MDN)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
10. [Cross-Origin Isolation Guide (web.dev)](https://web.dev/articles/cross-origin-isolation-guide)

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Roadmap Tasks

**All tasks are now tracked in the feature-log system.**

See:
- `feature-log/doppler/*.jsonl` - JSONL database of all features and tasks
- `/feature-log-query --status planned` - Query planned tasks
- `/feature-log-query --priority P0` - Query P0 tasks

---

## Technical Deep-Dives

For technical implementation details, see:
- `ARCHITECTURE.md` for subsystem breakdown and pipeline flow
- `CONFIG.md` for kernel path definitions and tuning knobs
- `FORMATS.md` for RDRR manifest fields and tensor layout

---

## Status Overview

For current operational status, test results, and recent fixes, see:
- [Postmortems](POSTMORTEMS.md) (Issue history)
- [Architecture](ARCHITECTURE.md) (System design)


## Target Models

Priority models to benchmark against WebLLM and establish DOPPLER's competitive position.

## Target Models

| Model | Size | Strategic Value | Architecture |
|-------|------|-----------------|--------------|
| **Llama-3.1-8B-Q4** | ~4.5GB | WebLLM's benchmark star - beat them here | LlamaForCausalLM |
| **Gemma-2-9B-Q4** | ~5GB | Gemma focus, builds on Gemma 3 work | Gemma2ForCausalLM |
| **Phi-3.5-mini-Q4** | ~2GB | Speed king (WebLLM claims 71 tok/s) | Phi3ForCausalLM |

## Architecture Notes

### Llama-3.1-8B
- Standard LLaMA architecture (well-supported)
- GQA (Grouped Query Attention): 8 KV heads for 32 attention heads
- RoPE with theta=500000
- SwiGLU FFN
- 32 layers, hidden_size=4096, intermediate=14336
- **Advantage**: Most widely tested architecture, good baseline

### Gemma-2-9B
- Similar to Gemma 3 but without sliding window
- GQA: 8 KV heads for 16 attention heads
- Logit soft-capping (30.0 for attention, 30.0 for final)
- Pre/post layernorm sandwich structure
- 42 layers, hidden_size=3584, intermediate=14336
- **Advantage**: Builds on our Gemma 3 work, shared norm offset handling

### Phi-3.5-mini
- Microsoft's efficient architecture
- Long context (128K) with rope scaling
- 32 layers, hidden_size=3072, intermediate=8192
- SwiGLU activation
- **Advantage**: Smallest, fastest - good for iteration speed

## Recommended Order

1. **Phi-3.5-mini-Q4** (Start here)
   - Smallest model = fastest iteration
   - Quick feedback loop for Q4K kernel tuning
   - Clear benchmark target (71 tok/s)
   - If we can't beat WebLLM here, we have work to do

2. **Llama-3.1-8B-Q4** (Second)
   - Most standard architecture
   - WebLLM's flagship benchmark
   - Direct competitive comparison

3. **Gemma-2-9B-Q4** (Third)
   - Leverages our Gemma expertise
   - Differentiator from WebLLM (they focus on Llama)

## HuggingFace Sources

```bash
# Phi-3.5-mini (already Q4 available)
huggingface-cli download microsoft/Phi-3.5-mini-instruct

# Llama-3.1-8B (need Q4 quantized or quantize ourselves)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Gemma-2-9B
huggingface-cli download google/gemma-2-9b-it
```

## Success Metrics

| Model | WebLLM Baseline | Target | Stretch |
|-------|-----------------|--------|---------|
| Phi-3.5-mini-Q4 | 71 tok/s | 75 tok/s | 90 tok/s |
| Llama-3.1-8B-Q4 | ~35 tok/s | 40 tok/s | 50 tok/s |
| Gemma-2-9B-Q4 | ~30 tok/s | 35 tok/s | 45 tok/s |

*Targets assume M1/M2 Mac with 16GB RAM*

## Conversion Commands

```bash
# After downloading, convert to RDRR format:
npx tsx src/converter/node-converter.js \
  ~/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/<hash>/ \
  models/phi-3.5-mini \
  --quantize q4_k_m

npx tsx src/converter/node-converter.js \
  ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<hash>/ \
  models/llama-3.1-8b \
  --quantize q4_k_m
```


## Browser Training

## Goal

Add LoRA fine-tuning capability to Doppler, enabling in-browser training of FunctionGemma and other small models without external infrastructure.

**Principle:** Doppler provides training primitives (backward kernels, optimizer steps). Reploid/gamma decides training policy (when to train, what data, loss thresholds).

---

## Status

Implemented in `src/training` with a runnable demo harness and test coverage.

Key entry points:
- Training engine overview: `../src/training/README.md`
- Demo harness: `app/training.html` + `app/training-harness.js`
- Training tests: `tests/training/browser/test-page.js` and `npm test -- --training`
- Export spec: `FORMATS.md`

This document now serves as a historical design reference.

---

## Why Browser Training

1. **Self-improvement loop:** Reploid can fine-tune FunctionGemma on its own traces without leaving the browser
2. **Privacy:** Training data never leaves the device
3. **No infra:** No Colab, no cloud GPUs, no Python environment required
4. **Unique capability:** No other browser ML framework offers LoRA training

---

## Target Model: FunctionGemma 270M

**Exact forward graph (ops and shapes for seq_len=128, batch=1):**

```
embed:        [vocab=256128, dim=1152] → [128, 1152]
per layer (×12):
  rmsnorm:    [128, 1152] → [128, 1152]
  qkv_proj:   [1152, 3456] → Q[128,1152], K[128,1152], V[128,1152]
  rope:       Q, K × freqs → Q', K'
  attention:  softmax(Q'K'^T / √d) × V → [128, 1152]
  o_proj:     [1152, 1152] → [128, 1152]
  residual:   + input
  rmsnorm:    [128, 1152] → [128, 1152]
  gate_proj:  [1152, 6912] → [128, 6912]
  up_proj:    [1152, 6912] → [128, 6912]
  silu:       gate × silu(up) → [128, 6912]
  down_proj:  [6912, 1152] → [128, 1152]
  residual:   + input
final_rmsnorm: [128, 1152] → [128, 1152]
lm_head:      [1152, 256128] → [128, 256128]
```

**Backward kernels required for this graph:**

| Op | Backward Kernel | Notes |
|----|-----------------|-------|
| embed | embed_backward | Uses scatter_add for gradient accumulation to embedding table |
| rmsnorm | rmsnorm_backward | Variance chain rule |
| matmul (all projs) | matmul_backward | dW = x.T @ dy, dx = dy @ W.T (reuses forward with transpose) |
| rope | rope_backward | sin/cos derivatives |
| attention | attention_backward | softmax + matmul chain, **includes causal mask backward** |
| softmax | softmax_backward | y * (dy - sum(y * dy)) |
| silu | silu_backward | σ(x) * (1 + x * (1 - σ(x))) |
| gelu | gelu_backward | GELU derivative (if model uses GELU variant) |
| scale | scale_backward | Identity scaled by constant (for LoRA alpha) |
| residual | (none) | Gradient passes through unchanged |
| cross_entropy | cross_entropy_backward | -log(p) derivative, returns logit gradients |

---

## Scope

### Phase 1: LoRA-Only Training

Train LoRA adapters while keeping base model frozen. This minimizes memory and avoids weight-gradient kernels for base model.

**In scope:**
- Autograd tape for forward graph (config-backed op registry)
- Backward kernels for all FunctionGemma ops (see table above)
- LoRA forward/backward through adapter layers
- Adam optimizer kernel
- Cross-entropy loss kernel
- Gradient clipping
- Mixed precision (f16 activations, f32 gradients)
- Adapter export (custom RDRR-LoRA format, not safetensors)

**Out of scope (Phase 2+):**
- Full fine-tuning (weight gradients for base model)
- Gradient checkpointing
- Per-expert MoE LoRA routing
- Distributed training across P2P mesh

**Implementation order:**

1. Autograd tape
2. Backward kernels
3. LoRA layer
4. Optimizer
5. Training loop
6. Data pipeline
7. Testing + parity

---

## Architecture

### Engine/Driver Split

```
┌─────────────────────────────────────────────────────────────┐
│  REPLOID (Driver) - Training Policy                         │
│  ├─ decideWhenToTrain(traces, threshold)                    │
│  ├─ prepareBatch(traces, batchSize)                         │
│  ├─ calculateLoss(logits, targets) → policy                 │
│  ├─ shouldCheckpoint(step, metrics)                         │
│  └─ exportAdapter(path)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  DOPPLER (Engine) - Training Primitives                     │
│  ├─ forward(input, weights, loraAdapters) → activations     │
│  ├─ backward(loss, tape) → gradients                        │
│  ├─ optimizerStep(params, grads, state) → updated           │
│  ├─ clipGradients(grads, maxNorm) → clipped                 │
│  ├─ crossEntropyLoss(logits, targets) → loss                │
│  └─ exportLoraWeights(adapters) → ArrayBuffer               │
└─────────────────────────────────────────────────────────────┘
```

### Memory Layout

**Realistic estimates for different configurations:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FunctionGemma 270M + LoRA r=16                                             │
│  Config: seq_len=128, batch=1 (MINIMUM)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB   (q,k,v,o × 12 layers)      │
│  Activations (forward, f16)              85 MB  (all intermediate states)  │
│  Activation gradients (backward, f32)    170 MB (2× activations for f32)   │
│  LoRA gradients (f32)                    8 MB                               │
│  Optimizer state (Adam m,v for LoRA)     16 MB  (2× LoRA params, f32)      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~823 MB                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Config: seq_len=256, batch=2 (REALISTIC)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB                               │
│  Activations (forward, f16)              340 MB (4× minimum)               │
│  Activation gradients (backward, f32)    680 MB (2× activations)           │
│  LoRA gradients (f32)                    8 MB                               │
│  Optimizer state (Adam)                  16 MB                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~1.6 GB                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Config: seq_len=512, batch=4 (AGGRESSIVE)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Base weights (frozen, f16)              540 MB                             │
│  LoRA adapters (trainable, f16)          4 MB                               │
│  Activations (forward, f16)              1.36 GB (16× minimum)             │
│  Activation gradients (backward, f32)    2.72 GB                            │
│  LoRA gradients + optimizer              24 MB                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  TOTAL                                   ~4.6 GB                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Hardware requirements:**

| Config | Min VRAM | Recommended |
|--------|----------|-------------|
| seq=128, batch=1 | 2GB | 4GB |
| seq=256, batch=2 | 4GB | 8GB |
| seq=512, batch=4 | 6GB | 16GB |

**Note:** Unified RAM Macs can use system memory, so 8GB M1 works for realistic config. Discrete GPUs need the VRAM directly.

---

## Config Wiring

Training config must follow Doppler's "config as code" pattern. This section defines the plumbing.

### Backward Registry

```
src/config/schema/backward-registry.js      # Schema definition
src/config/schema/backward-registry.d.ts
src/config/kernels/backward-registry.json   # Data (op→kernel mapping)
src/config/backward-registry-loader.js      # Loader with validation
src/config/backward-registry-loader.d.ts
```

**Loader integrates with config merge:**

```javascript
// src/config/backward-registry-loader.js
import { validateBackwardRegistry } from './schema/backward-registry.js';
import backwardRegistryData from './kernels/backward-registry.json';

export function loadBackwardRegistry() {
  const validated = validateBackwardRegistry(backwardRegistryData);
  return validated;
}
```

### Training Config Merge

Training defaults merge into Doppler config via `createTrainingConfig()`:

```javascript
// src/config/training-defaults.js
import { DEFAULT_DOPPLER_CONFIG } from './index.js';

export const DEFAULT_TRAINING_CONFIG = {
  ...DEFAULT_DOPPLER_CONFIG,
  training: {
    enabled: false,
    lora: {
      rank: 16,
      alpha: 32,
      dropout: 0.0,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    optimizer: {
      type: 'adam',
      lr: 2e-4,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.0,
    },
    gradient: {
      maxNorm: 1.0,
      accumSteps: 1,
    },
    precision: {
      activations: 'f16',
      gradients: 'f32',
      loraParams: 'f32',
    },
    attention: {
      recomputeForward: false,  // See "Forward Cache vs Recompute" below
    },
  },
};

export function createTrainingConfig(overrides = {}) {
  return mergeConfig(DEFAULT_TRAINING_CONFIG, overrides);
}
```

### Kernel Registry Wiring

Backward kernels live in `src/gpu/kernels/backward/` but the global kernel registry expects WGSL paths relative to `src/gpu/kernels/`.

**Registry entries for backward kernels (structure matches kernel registry schema):**

```json
// Addition to src/config/kernels/registry.json
{
  "operations": {
    "backward": {
      "description": "Training backward kernels",
      "baseBindings": [
        { "index": 0, "name": "uniforms", "type": "uniform" },
        { "index": 1, "name": "input", "type": "read-only-storage" },
        { "index": 2, "name": "output", "type": "storage" }
      ],
      "baseUniforms": {
        "size": 16,
        "fields": [
          { "name": "size", "type": "u32", "offset": 0 }
        ]
      },
      "variants": {
        "embed_backward": {
          "wgsl": "backward/embed_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [256, 1, 1],
          "requires": []
        },
        "matmul_backward": {
          "wgsl": "backward/matmul_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [16, 16, 1],
          "requires": []
        },
        "attention_backward": {
          "wgsl": "backward/attention_backward.wgsl",
          "entryPoint": "main",
          "workgroup": [256, 1, 1],
          "requires": []
        }
      }
    }
  }
}
```

**Wrappers reference registry paths:**

```javascript
// src/gpu/kernels/backward/matmul_backward.js
import { getKernelConfig } from '../utils.js';

const config = getKernelConfig('backward/matmul_backward');
// config.wgsl resolves to 'backward/matmul_backward.wgsl'
```

### Buffer Usage Constants

Use existing Doppler buffer usage constants, not ad-hoc WebGPU enums:

```javascript
// Use from src/gpu/buffer-pool.js
import { BufferUsage } from '../buffer-pool.js';

// BufferUsage.STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
// BufferUsage.STORAGE_READ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
// BufferUsage.UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
```

LoRA and gradient buffers use `BufferUsage.STORAGE` (read/write for optimizer updates).

---

## Forward Cache vs Recompute Decision

**Problem:** attention_backward needs softmax outputs from forward pass. Two options:

| Strategy | Memory | Compute | When to Use |
|----------|--------|---------|-------------|
| **Cache** | +340MB (seq=256, batch=2) | 1x | VRAM > 2GB available |
| **Recompute** | +0 | ~2x | Memory-constrained |

**Impact on memory estimates:**

```
seq=256, batch=2 with CACHE:
  Activations:     340 MB
  Attn weights:    340 MB  ← cached softmax outputs
  Gradients:       680 MB
  TOTAL:          ~1.9 GB  (was 1.6 GB)

seq=256, batch=2 with RECOMPUTE:
  Activations:     340 MB
  Attn weights:    0 MB    ← recomputed in backward
  Gradients:       680 MB
  TOTAL:          ~1.6 GB
```

**Decision:** Default to `recomputeForward: false` (cache) for seq≤256. Auto-switch to recompute if VRAM pressure detected. Configurable via `training.attention.recomputeForward`.

---

## Implementation Plan

### 1. Autograd Tape

**New files:**
- `src/training/autograd.js` - Tape implementation
- `src/training/autograd.d.ts` - Types
- `src/config/kernels/backward-registry.json` - Op→backward kernel mapping

**Responsibilities:**
- Track operations during forward pass
- Store references to inputs/outputs for backward
- Replay backward in reverse order using registry-defined kernels
- Handle in-place operations safely
- Integrate with buffer pool for gradient allocation

**Config-backed op registry (no ad-hoc strings):**

```json
// src/config/kernels/backward-registry.json
{
  "ops": {
    "embed": {
      "backward": "embed_backward",
      "grads": ["weight"],
      "notes": "Uses scatter_add for gradient accumulation"
    },
    "matmul": {
      "backward": "matmul_backward",
      "grads": ["input", "weight"],
      "requires_transpose": true
    },
    "rmsnorm": {
      "backward": "rmsnorm_backward",
      "grads": ["input", "gamma"]
    },
    "attention": {
      "backward": "attention_backward",
      "grads": ["q", "k", "v"],
      "notes": "Includes causal masking backward"
    },
    "softmax": {
      "backward": "softmax_backward",
      "grads": ["input"]
    },
    "silu": {
      "backward": "silu_backward",
      "grads": ["input"]
    },
    "gelu": {
      "backward": "gelu_backward",
      "grads": ["input"]
    },
    "rope": {
      "backward": "rope_backward",
      "grads": ["input"]
    },
    "scale": {
      "backward": "scale_backward",
      "grads": ["input"],
      "notes": "For LoRA alpha scaling"
    },
    "cross_entropy": {
      "backward": "cross_entropy_backward",
      "grads": ["logits"],
      "notes": "Loss function, returns scalar loss + logit gradients"
    }
  }
}
```

**Usage pattern (ops resolved from registry, not strings):**

```javascript
import { loadBackwardRegistry } from '../config/backward-registry-loader.js';

// Usage pattern - ops are registry keys, not magic strings
const tape = new AutogradTape(loadBackwardRegistry());
tape.watch(input);

const h1 = tape.record(OpType.MATMUL, matmul, [input, W1]);
const h2 = tape.record(OpType.RMSNORM, rmsnorm, [h1, gamma]);
const logits = tape.record(OpType.MATMUL, matmul, [h2, W_out]);

const loss = crossEntropy(logits, targets);
const grads = tape.backward(loss); // Uses registry to dispatch backward kernels
```

### 2. Backward Kernels

**New WGSL files (consistent `_backward` naming):**

| Kernel | File | Complexity | Notes |
|--------|------|------------|-------|
| embed_backward | `embed_backward.wgsl` | Medium | Uses scatter_add for sparse gradient accumulation |
| matmul_backward | `matmul_backward.wgsl` | Low | Reuse matmul with transpose |
| softmax_backward | `softmax_backward.wgsl` | Medium | |
| rmsnorm_backward | `rmsnorm_backward.wgsl` | Medium | Variance chain rule |
| attention_backward | `attention_backward.wgsl` | **High** | Causal mask + numerical stability (see below) |
| rope_backward | `rope_backward.wgsl` | Medium | sin/cos derivatives |
| gelu_backward | `gelu_backward.wgsl` | Low | |
| silu_backward | `silu_backward.wgsl` | Low | |
| scale_backward | `scale_backward.wgsl` | Low | For LoRA alpha/rank scaling |
| cross_entropy_backward | `cross_entropy_backward.wgsl` | Low | Loss gradient to logits |
| adam | `adam.wgsl` | Low | Optimizer step (not backward, but grouped here) |

**attention_backward detailed requirements (highest complexity kernel):**

1. **Causal mask backward:** Gradients must not flow through masked positions. Mask is applied before softmax, so backward must zero gradients for masked (future) positions.

2. **Numerical stability:** Forward uses `softmax(QK^T / √d - mask * 1e9)`. Backward must:
   - Recompute softmax outputs (or cache from forward)
   - Use stable softmax gradient: `dS = S * (dO - sum(S * dO))`
   - Handle the `1/√d` scaling in gradient flow

3. **Gradient outputs:** Must produce `dQ`, `dK`, `dV` from `dO` (output gradient):
   ```
   dV = softmax(QK^T)^T @ dO
   dS = dO @ V^T
   dQK = softmax_backward(dS, S)  # where S = softmax(QK^T/√d)
   dQ = dQK @ K / √d
   dK = dQK^T @ Q / √d
   ```

4. **Memory consideration:** May need to recompute attention weights rather than caching (memory vs compute tradeoff). Flash attention style recomputation could reduce memory at cost of ~2x compute.

**Complexity factors** to account for:
- attention_backward with causal masking is the highest complexity kernel
- embed_backward scatter_add path needs careful testing
- Each kernel needs .js wrapper + .d.ts types

**Key insight:** `matmul_backward` is just matmul with transposed operands:
```
dW = matmul(x.T, dy)  // gradient w.r.t. weights
dx = matmul(dy, W.T)  // gradient w.r.t. input
```

Existing `matmul.wgsl` variants can be reused with transpose flags.

### 3. LoRA Layer

**New files:**
- `src/training/lora.js` - LoRA adapter implementation
- `src/training/lora.d.ts` - Types
- `src/config/schema/lora.js` - LoRA config schema
- `src/config/schema/lora.d.ts`

**Buffer pool integration:** LoRA tensors use buffer pool with explicit dtype + usage flags. No per-step allocations.

```javascript
import { acquireBuffer, releaseBuffer, BufferUsage } from '../gpu/buffer-pool.js';
import { createTensor, tensorBytes } from '../gpu/tensor.js';
import { getTrainingConfig } from '../config/training-defaults.js';

// LoRA: h = Wx + α(BA)x where B, A are low-rank adapters
class LoraAdapter {
  /**
   * @param {LoraAdapterConfig} config - From training config schema
   */
  constructor(config) {
    const { inDim, outDim, rank, alpha } = config;
    const { loraParams: dtype } = getTrainingConfig().training.precision;

    // Acquire from buffer pool with explicit dtype + Doppler's BufferUsage constants
    const aBytes = tensorBytes([inDim, rank], dtype);
    const bBytes = tensorBytes([rank, outDim], dtype);

    this.A = createTensor(
      acquireBuffer(aBytes, BufferUsage.STORAGE),  // Uses Doppler's constant
      dtype,
      [inDim, rank],
      'lora_A'
    );
    this.B = createTensor(
      acquireBuffer(bBytes, BufferUsage.STORAGE),  // Uses Doppler's constant
      dtype,
      [rank, outDim],
      'lora_B'
    );
    this.alpha = alpha;
    this.rank = rank;
  }

  forward(x, tape) {
    const down = tape.record(OpType.MATMUL, matmul, [x, this.A]);
    const up = tape.record(OpType.MATMUL, matmul, [down, this.B]);
    // Scale by alpha/rank as per LoRA paper
    return tape.record(OpType.SCALE, scale, [up, this.alpha / this.rank]);
  }

  dispose() {
    releaseBuffer(this.A.buffer);
    releaseBuffer(this.B.buffer);
  }
}
```

**LoRA placement (which layers get adapters):**

| Layer | LoRA Target | Rationale |
|-------|-------------|-----------|
| q_proj | Yes | Attention query transformation |
| k_proj | Yes | Attention key transformation |
| v_proj | Yes | Attention value transformation |
| o_proj | Yes | Attention output |
| gate_proj | Optional | FFN gating |
| up_proj | Optional | FFN expansion |
| down_proj | Optional | FFN contraction |

Default: q, k, v, o only (per gamma's train.py). Configurable via schema.

### 4. Optimizer Kernel

**New files:**
- `src/gpu/kernels/adam.wgsl`
- `src/training/optimizer.js`

```wgsl
// adam.wgsl - ~50 lines
@compute @workgroup_size(256)
fn adam_step(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let i = gid.x;
  let g = grads[i];

  // Update biased first moment
  m[i] = beta1 * m[i] + (1.0 - beta1) * g;

  // Update biased second moment
  v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

  // Bias correction
  let m_hat = m[i] / (1.0 - pow(beta1, f32(step)));
  let v_hat = v[i] / (1.0 - pow(beta2, f32(step)));

  // Update parameters
  params[i] = params[i] - lr * m_hat / (sqrt(v_hat) + eps);
}
```

### 5. Training Loop Integration

**New files:**
- `src/training/trainer.js` - High-level training API
- `src/training/index.js` - Exports

```javascript
// Doppler training primitive
async function trainStep(model, batch, config) {
  const tape = new AutogradTape();

  // Forward
  const logits = model.forward(batch.input, tape);
  const loss = crossEntropyLoss(logits, batch.targets);

  // Backward
  const grads = tape.backward(loss);

  // Clip
  const clipped = clipGradients(grads, config.maxNorm);

  // Update
  optimizer.step(model.loraParams(), clipped);

  return { loss: await loss.read() };
}
```

### 6. Data Pipeline

**New files:**
- `src/training/dataloader.js` - Batch preparation
- `src/training/checkpoint.js` - IndexedDB persistence

```javascript
// Browser-native data loading
class DataLoader {
  constructor(dataset, batchSize, shuffle = true) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
  }

  async *batches() {
    const indices = this.shuffle ? shuffled(this.dataset.length) : range(this.dataset.length);
    for (let i = 0; i < indices.length; i += this.batchSize) {
      const batch = indices.slice(i, i + this.batchSize);
      yield this.collate(batch);
    }
  }
}
```

### 7. Testing

**Test categories:**

| Category | Tests |
|----------|-------|
| Per-kernel numerical | Each backward kernel vs PyTorch reference |
| Gradient checking | Finite difference verification |
| End-to-end parity | Full training loop vs HuggingFace TRL |
| Memory stability | No leaks over 1000 steps |
| Precision | f16/f32 mixed precision correctness |

**New test files:**
- `tests/training/autograd.test.js`
- `tests/training/backward-kernels.test.js`
- `tests/training/lora.test.js`
- `tests/training/optimizer.test.js`
- `tests/training/e2e-parity.test.js`

---

## File Structure

**Style compliance:** Every `.js` has a matching `.d.ts`. All defaults in schema/config.

```
doppler/src/training/
├── index.js              # Exports
├── index.d.ts            # Types
├── autograd.js           # Tape implementation
├── autograd.d.ts
├── lora.js               # LoRA adapter
├── lora.d.ts
├── optimizer.js          # Adam wrapper
├── optimizer.d.ts
├── dataloader.js         # Batch preparation
├── dataloader.d.ts
├── checkpoint.js         # IndexedDB persistence
├── checkpoint.d.ts
├── trainer.js            # High-level training primitives
└── trainer.d.ts

doppler/src/config/schema/training.js         # Training config schema
doppler/src/config/schema/training.d.ts
doppler/src/config/schema/lora.js             # LoRA config schema
doppler/src/config/schema/lora.d.ts
doppler/src/config/schema/backward-registry.js    # Backward registry schema
doppler/src/config/schema/backward-registry.d.ts
doppler/src/config/training-defaults.js       # Default values (NOT in runtime code)
doppler/src/config/training-defaults.d.ts
doppler/src/config/backward-registry-loader.js    # Loader with validation
doppler/src/config/backward-registry-loader.d.ts

doppler/src/config/kernels/backward-registry.json # Backward op→kernel mapping
doppler/src/config/kernels/registry.json          # (existing, add training category entries)

doppler/src/gpu/kernels/
├── (existing forward kernels)
├── backward/                     # NEW directory for backward kernels
│   ├── embed_backward.wgsl       # Uses scatter_add internally
│   ├── embed_backward.js
│   ├── embed_backward.d.ts
│   ├── matmul_backward.wgsl
│   ├── matmul_backward.js
│   ├── matmul_backward.d.ts
│   ├── softmax_backward.wgsl
│   ├── softmax_backward.js
│   ├── softmax_backward.d.ts
│   ├── rmsnorm_backward.wgsl
│   ├── rmsnorm_backward.js
│   ├── rmsnorm_backward.d.ts
│   ├── attention_backward.wgsl   # Includes causal mask handling
│   ├── attention_backward.js
│   ├── attention_backward.d.ts
│   ├── rope_backward.wgsl
│   ├── rope_backward.js
│   ├── rope_backward.d.ts
│   ├── silu_backward.wgsl
│   ├── silu_backward.js
│   ├── silu_backward.d.ts
│   ├── gelu_backward.wgsl
│   ├── gelu_backward.js
│   ├── gelu_backward.d.ts
│   ├── scale_backward.wgsl       # For LoRA alpha scaling
│   ├── scale_backward.js
│   ├── scale_backward.d.ts
│   ├── cross_entropy_backward.wgsl
│   ├── cross_entropy_backward.js
│   ├── cross_entropy_backward.d.ts
│   ├── adam.wgsl
│   ├── adam.js
│   ├── adam.d.ts
│   ├── index.js                  # Backward kernel exports
│   └── index.d.ts
```

---

## Verification Criteria

Phase 1 complete when:

1. [ ] FunctionGemma 270M LoRA trains in Chrome on M1 Mac (8GB)
2. [ ] Loss decreases on toy dataset (overfitting test)
3. [ ] Exported RDRR-LoRA adapter loads in Doppler inference
4. [ ] Per-kernel tests pass against PyTorch reference
5. [ ] No memory leaks over 1000 training steps
6. [ ] Training speed: >5 samples/sec on M1 (seq=128, batch=1)

**Note on export interop:** RDRR-LoRA is Doppler-native. For Ollama/llama.cpp interop, a separate `rdrr-to-gguf` converter would be needed (out of scope for Phase 1, could be Phase 1.5).

---

## Future Phases

### Phase 2: Full Fine-tune
- Weight gradient kernels for base model
- Gradient checkpointing for memory efficiency
- Larger optimizer state buffers

### Phase 3: Per-Expert MoE LoRA
- Expert-aware LoRA routing
- Per-expert adapter storage
- Dynamic LoRA loading based on router decisions

### Phase 4: Distributed Training
- Gradient aggregation over P2P mesh
- Federated averaging for privacy
- Expert sharding across peers

---

## Dependencies

**Existing Doppler infrastructure used:**
- `buffer-pool.js` - Gradient buffer allocation
- `tensor.js` - Tensor abstraction with dtype
- `kernel-runtime.js` - Kernel dispatch
- `profiler.js` - Performance monitoring
- `device.js` - WebGPU device management

**No new external dependencies required.**

---

## Open Questions

1. **Loss scaling:** Do we need dynamic loss scaling for f16, or is static sufficient for 270M?
   - *Recommendation:* Start with static scaling (1.0), add dynamic if instability observed.

2. **Reploid integration:** Should TraceStore auto-trigger training, or manual only?
   - *Recommendation:* Manual first, auto-trigger as Phase 1.5 enhancement.

3. **LoRA param quantization:** Train in f16 or f32?
   - *Recommendation:* Train in f32 for stability, quantize to f16 on export.

---

## Decisions Made

1. **Export format:** Custom RDRR-LoRA extension, not safetensors.
   - Safetensors requires external dependency or custom writer.
   - RDRR-LoRA keeps everything in Doppler's existing format ecosystem.
   - **Interop path (Phase 1.5):** Add `rdrr-to-gguf` converter for Ollama/llama.cpp if needed.

2. **Max batch size for 8GB:** seq=256, batch=2 is the practical limit (~1.6GB).
   - Leaves headroom for browser overhead and other tabs.

3. **Attention backward includes causal masking.**
   - FunctionGemma uses causal attention; backward must handle mask gradients correctly.
   - Not a separate kernel; integrated into attention_backward.wgsl.

---

## Minimal E2E Training Target

**Toy dataset for validation:**
- 100 FunctionGemma tool-call traces from gamma's training format
- Known loss curve: should reach <0.5 cross-entropy in 50 steps (overfitting test)
- If loss doesn't decrease, backward kernels have bugs

**Parity test:**
- Same 100 traces through gamma's Python train.py
- Compare loss curves step-by-step
- Tolerance: <5% relative difference at each step

**Performance target:**
- >5 samples/sec on M1 Mac (8GB) at seq=128, batch=1
- >2 samples/sec at seq=256, batch=2

---

*Created: January 2026*
*Status: Implemented (engine primitives)*
*Last updated: January 2026 (loss + clipping + adapter export)*


## FunctionGemma Refactor

## Goal

Enforce strict Engine (Doppler) vs Driver (Reploid) separation for FunctionGemma multi-model orchestration.

**Principle:** Doppler never decides, it only executes. Reploid passes policy decisions as parameters to Doppler's primitives.

## Current Violations

| Feature | Current Location | Correct Location | Violation |
|---------|------------------|------------------|-----------|
| Prompt templates | `multi-model-network.js:352-365` | Reploid | Hardcoded "Generate code...", "Review this code..." |
| Temporal ring loop | `multi-model-network.js:285-341` | Reploid | Seed/Reflect/Refine orchestration |
| UCB1 selection | `functiongemma.js` | Reploid | Exploration/exploitation policy |
| Evolution/GA | `functiongemma.js` | Reploid | `runEvolution`, mutation, crossover |
| Arena competition | `functiongemma.js` | Reploid | `runArena`, head-to-head |
| Fitness scoring | `functiongemma.js` | Reploid | `calculateBaseFitness` heuristics |
| Task classification | `functiongemma.js` | Reploid | Keyword matching for routing |

## Architectural Split

### Doppler (Engine) - KEEPS

Primitives that execute without making policy decisions:

```
executeExpert(id, prompt, options) → token_stream
setSharedPrefix(prompt) → KVCacheSnapshot
executeChain(expertIds, prompt) → string[]
executeParallel(tasks) → Record<string, string>
combineOutputs(outputs, combinerConfig) → string
executeGenome(genome, prompt) → string
mergeLogits(buffers, weights) → GPUBuffer  [NEW]
sampleFromMerged(logits, params) → tokenId  [NEW]
```

### Reploid (Driver) - OWNS

All orchestration, policy, and decision-making:

```
buildTemporalPrompt(task, turn, role) → string
executeTemporalSelfRing(task, config) → result
evolveTopology(tasks, config) → NetworkGenome
runArenaEvolution(tasks, config) → winner
selectExpertUCB1(taskType, candidates) → expertId
calculateFitness(output, task) → score
classifyTask(description) → taskType
```

## Files Changed

### Doppler - Remove Orchestration

| File | Action |
|------|--------|
| `src/inference/multi-model-network.js` | Remove `executeTemporalRing`, `buildTemporalPrompt`, `detectTemporalConvergence` |
| `src/inference/multi-model-network.d.ts` | Remove temporal ring types |
| `src/inference/functiongemma.js` | Gut to thin re-export of primitives |
| `src/inference/functiongemma.d.ts` | Primitive types only |
| `src/inference/functiongemma.ts.wip` | Delete (superseded by Reploid) |

### Doppler - Add GPU Primitives

| File | Purpose |
|------|---------|
| `src/gpu/kernels/logit-merge.js` | GPU tensor merging for multi-model ensemble |

### Documentation

| File | Change |
|------|--------|
| `ARCHITECTURE.md` | Add "Engine vs Driver Boundary" section |
| `../../reploid/docs/design/FUNCTIONGEMMA.md` | Clarify Doppler provides primitives only |
| `../../ARCHITECTURE.md` | Add cross-project boundary definition |

## GPU Multi-Model Primitives

When multiple models are loaded, these operations MUST stay on GPU:

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU VRAM                            │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ Model A     │  │ Model B     │                          │
│  │ Logits      │  │ Logits      │                          │
│  └──────┬──────┘  └──────┬──────┘                          │
│         └───────┬────────┘                                  │
│                 ▼                                           │
│        ┌───────────────┐                                    │
│        │ mergeLogits() │  ← Doppler GPU primitive           │
│        └───────┬───────┘                                    │
│                ▼                                           │
│        ┌───────────────┐                                    │
│        │ sample()      │  ← Doppler GPU primitive           │
│        └───────┬───────┘                                    │
└────────────────┼────────────────────────────────────────────┘
                 ▼
            Token ID (CPU) → Reploid decides next action
```

**Reploid decides:** Which models, what weights, when to merge
**Doppler executes:** The actual tensor operations

## Verification

After refactor:
1. `npm test` passes in Doppler
2. Reploid's `functiongemma-orchestrator.js` works with cleaned Doppler primitives
3. No hardcoded prompts remain in Doppler
4. No loop/evolution logic remains in Doppler

---

*Created: January 2026*


## Competitive Analysis

**DOPPLER** (Distributed Object Parallel Processing Layer Executing REPLOID) is a browser-native LLM inference engine. It is part of the REPLOID system (Recursive Evolution Protocol Loop Orchestrating Inference DOPPLER), with model distribution handled by RDRR (Recursive DOPPLER Runtime Registry).
See also: `ARCHITECTURE.md`

## TL;DR

DOPPLER is a browser-native LLM inference engine using custom WebGPU (WGSL) kernels (no compiler like TVM). Key differentiators:

1. **Flash Attention in WGSL** - No other browser framework implements tiled Flash Attention directly in WGSL
2. **Custom MoE routing** - Direct WGSL scatter-add (vs TVM-generated gather) for lower VRAM overhead
3. **60GB model support (theoretical)** - Tiered memory system for unified memory architectures
4. **Native Bridge** - mmap access to local files, bypassing OPFS limits

**Caveat:** Performance benchmarks pending. WebLLM supports MoE (Mixtral) via TVM. DOPPLER must prove better performance.
See: `style/BENCHMARK_STYLE_GUIDE.md` and `TESTING.md`.

---

## Roadmap and Metrics

This document focuses on competitor context and technical constraints.

Implementation work, task tracking, and priorities live in the feature-log system
(`feature-log/doppler/*.jsonl`).

Benchmark and testing specs:

- `style/BENCHMARK_STYLE_GUIDE.md` (pipeline and system benchmarks, result schema)
- `TESTING.md` (kernel and segment tests, how to interpret correctness)

Key success metrics (the minimum needed for credible comparisons):

- Time to first token (cold and warm).
- Decode throughput (warm): tokens per second for greedy decode on a fixed workload set.
- Peak VRAM and readback bytes per token (logits and debug reads).
- MoE path correctness and performance (Mixtral and GPT-OSS).
- Model coverage matrix with VRAM requirements.

---

## Browser LLM Frameworks (Dec 2025)

| Framework | Compiler | Max Model | MoE | Flash Attn | Maturity | Community |
|-----------|----------|-----------|-----|------------|----------|-----------|
| **[WebLLM](https://github.com/mlc-ai/web-llm)** | TVM/MLC | ~31GB VRAM | **Yes** | Via TVM | Production | 16.9k stars |
| **[Transformers.js](https://huggingface.co/docs/transformers.js)** | ONNX Runtime | **4GB hard** | No | No | Production | 1.4M monthly users |
| **[MediaPipe LLM](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/web_js)** | TFLite | ~27GB | No | Unknown | Production | Google-backed |
| **[WeInfer](https://dl.acm.org/doi/10.1145/3696410.3714553)** | Custom | ~8GB | Unknown | Unknown | Research | Academic |
| **DOPPLER** | None (raw WGSL) | Claims 60GB | **Yes (custom)** | **Yes (custom)** | Prototype | n/a |

**Key gap (Dec 2025):** WebLLM supports Mixtral, but DOPPLER aims to push larger models via shard-based distribution (RDRR) and Native Bridge.

### WebLLM (MLC-AI)

The incumbent. Uses Apache TVM compiler for optimized WebGPU kernels.

> "Evaluations show that WebLLM can retain up to 80% native performance on the same device, with room to further close the gap."
>
> Source: [WebLLM Paper, arXiv 2412.15803](https://arxiv.org/abs/2412.15803), Dec 2024

**Model Catalog (Dec 2025):**

| Model | Params | Quantization | VRAM | Context |
|-------|--------|--------------|------|---------|
| Llama-3.2-1B-Instruct | 1B | q4f16 | ~1GB | 4k |
| Llama-3.2-3B-Instruct | 3B | q4f16 | ~2GB | 4k |
| Llama-3.1-8B-Instruct | 8B | q4f16 | ~5GB | 4k/128k |
| Llama-3-70B-Instruct | 70B | q4f16 | ~31GB | 4k |
| Qwen2.5-Coder-7B | 7B | q4f16 | ~5GB | 4k |
| DeepSeek-R1-Distill-Qwen-7B | 7B | q4f16 | ~5GB | 4k |
| Phi-3.5-vision-instruct | 4B | q4f16 | ~3GB | 4k |
| Gemma-2-9B | 9B | q4f16 | ~6GB | 4k |
| SmolLM2-1.7B | 1.7B | q4f16 | ~1GB | 4k |

Source: [WebLLM GitHub #683](https://github.com/mlc-ai/web-llm/issues/683), Dec 2025

**Notable missing models (as of Dec 2025):**
- **Gemma 3** (all variants) - DOPPLER's opportunity
- **Llama 3.3** - Latest Llama release
- **Phi-4** - Latest Microsoft model

**Quantization formats:** q4f16 (4-bit weights, f16 compute), q4f32, q0f16, q0f32

**API:** OpenAI-compatible

**Roadmap/WIP:**
- Function calling (tools API) - in progress
- Custom model compilation - available

MoE support: Mixtral support is reported. Verify current WebLLM catalog and hardware requirements.

### WeInfer (ACM Web Conference 2025) - Research Artifact

**Status (Dec 2025):** Stale research artifact. Last commit February 2025, based on WebLLM 0.2.46 (current is 0.2.80).

> "Evaluations across 9 different LLMs and 5 heterogeneous devices show that WeInfer delivers substantial improvements in decoding speed, achieving up to a 3.76x performance boost compared with WebLLM."
>
> Source: [ACM WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714553), April 2025

**Performance claims (validated):**
- 3.76x faster decode vs WebLLM v0.2.46
- Tested: 9 LLMs, 5 devices (RTX 4090, Apple M2, Windows GPUs)

**Key techniques (implement from scratch, don't use stale code):**

| Technique | Description | DOPPLER Status |
|-----------|-------------|----------------|
| **Buffer Reuse** | Pre-allocate fixed pool, reuse across ops | Partial (buffer-pool.js) |
| **Async Pipeline** | Decouple buffer prep from kernel dispatch | Not implemented |
| **Deferred Readback** | Batch GPU→CPU transfers, read only when needed | Not implemented |

**Threat assessment:**
- **WeInfer itself: Low threat** - abandoned research artifact
- **The techniques: High value** - 3.76x gains are real and reproducible
- **Real threat:** WebLLM adopting these optimizations
- DOPPLER should implement buffer reuse + async pipeline from first principles

**Resources:**
- Paper: [ACM WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714553)
- OpenReview: [openreview.net/forum?id=Qu2itILaoZ](https://openreview.net/forum?id=Qu2itILaoZ)
- Repo (stale): [github.com/csAugust/WeInfer](https://github.com/csAugust/WeInfer)

---

## Performance Benchmarks (December 2025)

Concrete tokens-per-second measurements from validated sources.

### WebLLM Official Benchmarks

From [WebLLM arXiv paper](https://arxiv.org/abs/2412.15803) on **M3 Max MacBook Pro**:

| Model | WebLLM (WebGPU) | MLC-LLM (Native) | % of Native |
|-------|-----------------|------------------|-------------|
| Llama-3.1-8B (q4) | **41.1 tok/s** | 57.7 tok/s | 71.2% |
| Phi-3.5-mini 3.8B (q4) | **71.1 tok/s** | 89.3 tok/s | 79.6% |
| 4-bit 3B model (generic) | **90 tok/s** | ~112 tok/s | ~80% |

From [MLC Blog](https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine):
- WebGPU preserves **up to 85% of native Metal** performance
- Test config: 64 prefill tokens, 128 decode tokens, 4-bit quantized

### Apple Silicon Performance (Native Inference)

From [GPU Benchmarks on LLM Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) with **Llama 3 8B Q4_K_M**:

| Hardware | Memory BW | Decode Speed |
|----------|-----------|--------------|
| M1 Max | 400 GB/s | 34.49 tok/s |
| M2 Ultra | 800 GB/s | 76.28 tok/s |
| M3 Max | 400 GB/s | 50.74 tok/s |

From [Ominous Industries](https://ominousindustries.com/blogs/ominous-industries/apple-silicon-speed-test-localllm-on-m1-vs-m2-vs-m2-pro-vs-m3) with **Llama 3 8B Q4**:

| Hardware | RAM | Speed | Notes |
|----------|-----|-------|-------|
| M1 Mac Mini | 16GB | ~10 tok/s | Baseline |
| M2 MacBook Air | 8GB | ~1 tok/s | **RAM-limited** |
| M2 Pro Mac Mini | 16GB | ~27 tok/s | |
| M3 iMac | 16GB | ~13 tok/s | |

From [Medium/Google Cloud](https://medium.com/google-cloud/gemma-3-performance-tokens-per-second-in-lm-studio-vs-ollama-mac-studio-m3-ultra-7e1af75438e4) with **M3 Ultra**:

| Model | LM Studio | Ollama |
|-------|-----------|--------|
| Gemma 3 1B | 237 tok/s | 182 tok/s |
| Gemma 3 4B | 134 tok/s | 103 tok/s |
| Gemma 3 27B | 33 tok/s | 25 tok/s |

### NVIDIA GPU Performance (Native Inference)

From [GPU Benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) and [CloudRift](https://www.cloudrift.ai/blog/benchmarking-rtx-gpus-for-llm-inference):

| GPU | VRAM | Llama 3 8B Q4 | Llama 3 8B F16 |
|-----|------|---------------|----------------|
| RTX 3090 | 24GB | 111.74 tok/s | 46.51 tok/s |
| RTX 4090 | 24GB | 127.74 tok/s | 54.34 tok/s |
| RTX 5090 | 32GB | **213 tok/s** | - |
| H100 | 80GB | 144 tok/s | - |
| A100 | 80GB | 138 tok/s | - |

### Browser-Specific Observations

From [Simon Willison](https://simonwillison.net/tags/webgpu/):

| Hardware | Browser | Model | Speed |
|----------|---------|-------|-------|
| M2 Mac 64GB | Chrome Canary | Llama 2 70B | **3.25 tok/s** |
| Generic | Chrome + WebGPU | Llama-3.2-1B | **~10 tok/s** |

From [Chrome Developer Blog](https://developer.chrome.com/blog/io24-webassembly-webgpu-2) - **WebGPU Optimizations**:

| Optimization | Model | Speedup |
|--------------|-------|---------|
| f16 vs f32 (prefill) | Llama 3 8B | **2.1x faster** |
| f16 vs f32 (decode) | Llama 3 8B | **1.3x faster** |
| Subgroups (Intel GPU) | Llama 3 8B prefill | **2.5x faster** |
| WebGPU vs WASM | Embedding benchmark | **30-120x faster** |

### WeInfer Performance Claims

From [ACM WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714553):
- **3.76x faster decode** vs WebLLM v0.2.46
- Tested: 9 LLMs, 5 devices (RTX 4090, Apple M2, Windows GPUs)
- Key techniques: buffer reuse, async pipeline, deferred readback

### DOPPLER Current Performance (December 2025)

**Model tested:** Gemma 1B Q4_K_M on M3 (macOS, Chrome 143)

| Metric | DOPPLER | Notes |
|--------|---------|-------|
| Decode speed | **4-5 tok/s** | Current (gemma-1b-q4) |
| Prefill speed | 36-83 tok/s | Varies by prompt length |
| GPU submits/decode | ~4 per token | Target: 1 |
| VRAM peak | ~2-4 GB | |

**Problem:** WebLLM benchmarks are on 8B models, not 1B. Scaling expectation:
- 1B model should be ~8x faster than 8B model
- If WebLLM gets 41 tok/s on 8B → expect ~300+ tok/s on 1B
- DOPPLER at 5 tok/s is **60x slower than expected**

### Key Insights for DOPPLER

| Metric | WebLLM Reference | DOPPLER Target | Current Gap |
|--------|------------------|----------------|-------------|
| 8B Q4 decode | 41 tok/s (M3 Max) | 40+ tok/s | Not tested yet |
| 1B Q4 decode | ~300+ tok/s (scaled) | 100+ tok/s | **5 tok/s (60x gap)** |
| Native ceiling | 57.7 tok/s (8B) | 80-85% | Unknown |
| Submits/token | 1 | 1 | 4 (4x too many) |

**Root causes under investigation:**
1. Submit amplification (4+ submits/token vs target of 1)
2. Missing buffer reuse (WeInfer technique)
3. Missing async pipeline (WeInfer technique)
4. Debug readbacks in hot path

**Reading speed reference:** 25-30 tok/s is considered above human reading speed.

### User-Collected Benchmarks (December 2025)

**Hardware:** MacBook M3 Air 24GB, Chrome

| Framework | Model | Prefill | Decode | Notes |
|-----------|-------|---------|--------|-------|
| **WebLLM** | gemma-2-9b-it-q4f16_1-MLC | 42.5 tok/s | **11.2 tok/s** | chat.webllm.ai |
| **Doppler** | gemma-3-1b-it-q4 | TBD | TBD | Needs testing |
| **MediaPipe** | Gemma-3-1B-LiteRT | TBD | TBD | Access gated |

**Benchmark prompt:** `The color of the sky is`

**Key observations:**
- WebLLM decode speed (11.2 tok/s) is the baseline to beat on M3 Air
- MediaPipe models transitioning from Kaggle to HuggingFace `litert-community` - currently gated
- Doppler 1B model should theoretically be faster than WebLLM 9B model

---

### Transformers.js (Hugging Face)

Largest browser ML community. Broad model support via ONNX Runtime Web, but hard 4GB limit.

**Scale (Oct 2025):**
- **1.4 million unique monthly users**
- **155 supported architectures**
- WebGPU mode: up to **100x faster** than WASM

Source: [JSNation 2025 Talk](https://gitnation.com/contents/transformersjs-state-of-the-art-machine-learning-for-the-web), [Transformers.js v3 Blog](https://huggingface.co/blog/transformersjs-v3), Oct 2024

> "Currently, there is no way for ONNX Runtime Web to run models larger than 4GB... WebAssembly has a memory limit of 4GB. This is the maximum amount of memory that a WebAssembly module can access because of the 32-bit addressing."
>
> Source: [ONNX Runtime Docs](https://onnxruntime.ai/docs/tutorials/web/large-models.html), Dec 2025

**Quantization:** fp32, fp16, q8 (default WASM), q4

**Notable demos:** SmolVLM (multimodal), Phi-3.5-WebGPU, Whisper-WebGPU

**Roadmap:**
- WebNN integration - in progress
- More architectures - ongoing (155→?)
- **WASM64 or direct GPU loading** - "may support in future" (would remove 4GB limit)

**Threat if 4GB limit removed:** Instant access to larger models for 1.4M users

### Google MediaPipe LLM

Google's official solution with custom workarounds for browser limits.

> "MediaPipe's earlier web APIs made heavy use of JavaScript primitives like ArrayBuffer when loading data, but many of these cannot support sizes past ~2GB. For the initial web LLM launch, they worked around the 2GB limitation by creating custom data copying routines... Google has since redesigned the model loading system to run much larger models like Gemma 1.1 7B. This 8.6GB model comprising 7 billion parameters is several times larger than any model they've run in a browser previously."
>
> Source: [Google AI Blog](https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/), 2024

**Model Catalog (Dec 2025):**

| Model | Params | Multimodal | Notes |
|-------|--------|------------|-------|
| Gemma-3n E2B | 2B | Image + Audio | Latest (Dec 2025) |
| Gemma-3n E4B | 4B | Image + Audio | Latest (Dec 2025) |
| Gemma 2B | 2B | No | Original |
| Gemma 4B | 4B | No | |
| Gemma 12B | 12B | No | |
| Gemma 27B | 27B | No | Largest |
| MedGemma-27B-Text | 27B | No | Medical domain |
| Phi-2 | 2.7B | No | Non-Google |
| Falcon-1B | 1B | No | Non-Google |
| StableLM-3B | 3B | No | Non-Google |

Source: [MediaPipe Web Guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/web_js), Dec 2025

**LoRA support:** Gemma-2 2B, Gemma 2B, Phi-2

**Key insight:** Google solved 2GB ArrayBuffer limit with custom data copying routines.

**Limitation:** Primarily Gemma-focused, limited non-Google model support.

**Key insight (Dec 2025):** MediaPipe uses **LiteRT (WASM-based)**, NOT pure WebGPU. This may create a performance gap vs WebGPU-native frameworks like DOPPLER and WebLLM.

**Access status (Dec 2025):**
- Models transitioning from Kaggle to HuggingFace `litert-community`
- Models are **gated** - require HuggingFace approval
- New LiteRT-LM format (`.litertlm`) replacing `.task` files
- Demo: https://mediapipe-studio.webapps.google.com/studio/demo/llm_inference

---

## Doppler Competitive Advantages (December 2025)

| Advantage | vs WebLLM | vs MediaPipe | vs Transformers.js |
|-----------|-----------|--------------|-------------------|
| **Gemma 3 support** | They don't have it | Gated access | Crashes (#1469) |
| **Pure WebGPU** | Draw (both WebGPU) | LiteRT uses WASM | Draw |
| **No access gates** | Draw | HF approval needed | Draw |
| **Custom WGSL kernels** | TVM black-box | LiteRT runtime | ONNX ops |
| **MoE support** | Both have it | No MoE | No MoE |
| **Flash Attention** | TVM-compiled | Unknown | No |

---

## Model Size Constraints

### WebGPU Buffer Limits (The Real Bottleneck)

> "Safari's Metal backend imposes a 256MB default buffer size limit on iPhone 6 devices, scaling up to only 993MB on iPad Pro, while Chrome's maxStorageBufferBindingSize is often limited to 128MB despite reporting higher capabilities."
>
> Source: [WebGPU Bugs Article](https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca)

| Browser | Buffer Limit | Notes |
|---------|--------------|-------|
| Chrome | ~128MB | Often lower than reported |
| Safari (iPhone) | 256MB | Metal backend |
| Safari (iPad Pro) | 993MB | Better but still limited |
| Firefox | Varies | WebGPU in v141+ |

### Practical Model Sizes

> "Currently, models in the 1-8 billion parameter range are most practical with quantization. Larger models may run on powerful devices, but memory and latency make them less user-friendly in browser environments."
>
> Source: [AI Competence Guide](https://aicompetence.org/ai-in-browser-with-webgpu/)

| Model Size | VRAM Required | Browser Feasibility |
|------------|---------------|---------------------|
| 1-3B (INT4) | 1-2GB | Good |
| 7-8B (INT4) | 4-6GB | Marginal |
| 13B+ | 8GB+ | Challenging |
| MoE (Mixtral 8x7B) | 90GB | Requires expert swapping |

### The 35% Compatibility Problem

> "WebGPU's promise of democratizing AI through browser-based LLM inference remains tantalizingly close yet frustratingly unattainable due to implementation bugs and ecosystem fragmentation. While WebLLM demonstrates that browser-based inference can achieve 80% of native performance, the 20% performance gap combined with compatibility issues affecting 35% of users, memory limitations preventing large model deployment, and platform-specific bugs requiring extensive workarounds creates an environment where production deployment remains impractical for most use cases."
>
> Source: [Medium Analysis](https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca)

---

## MoE Support Comparison (Dec 2025)

**DOPPLER's key differentiator:** Custom routing kernels (direct WGSL vs TVM compilation), and potential for demand-paging experts from disk (Native Bridge).

| Framework | MoE Status | Models | Implementation |
|-----------|------------|--------|----------------|
| **DOPPLER** | **Yes (GPU-native)** | Any (theoretical) | Custom WGSL topk + scatter_add |
| **WebLLM** | Yes | Mixtral 8x7B | TVM-generated kernels |
| **Transformers.js** | No | n/a | 4GB limit blocks Mixtral |
| **MediaPipe** | No | n/a | Dense models only |
| **vLLM (server)** | Yes | DeepSeek-V3, Mixtral, Qwen3 | CUDA/FlashInfer |

### Why MoE Matters

> "Since early 2025, nearly all leading frontier models use MoE designs."
>
> Source: [NVIDIA MoE Blog](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/), 2025

Top MoE models (server-side only as of Dec 2025):
- DeepSeek-V3 (671B params, 37B active)
- Mixtral 8x7B (46.7B params, 12.9B active)
- Qwen3 MoE variants
- Llama 4 Scout (109B params)

Source: [Red Hat vLLM+DeepSeek](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-and-sparse-moe-models-vllm-llm-d), Sept 2025

### DOPPLER MoE Validation Needed

- [ ] Run actual MoE model (e.g., Mixtral-instruct) end-to-end
- [ ] Benchmark expert swapping latency
- [ ] Compare vs WebLLM Mixtral on identical hardware

---

## JavaScript vs WASM Orchestration

A key architectural difference often overlooked: **what runs the non-GPU logic?**

### WebLLM: WASM Orchestration

WebLLM compiles orchestration logic (layer loops, tokenization, sampling) into WASM:

```
WebLLM Runtime:
├─ model.wasm
│   ├─ TVM-generated WGSL shaders (GPU kernels)
│   └─ Compiled C++ (orchestration, tokenization, sampling)
└─ model_weights.bin
```

> "The WASM library contains both compute kernels in WGSL (e.g. prefill, decode) and non-kernel functions in WebAssembly (e.g. BNFGrammar for JSON mode)."
>
> Source: [MLC Blog](https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine)

### DOPPLER: JavaScript Orchestration

DOPPLER uses JavaScript for all non-GPU logic:

```
DOPPLER Runtime:
├─ doppler-runtime.js
│   ├─ Hand-written WGSL shaders (GPU kernels)
│   └─ JavaScript (orchestration, tokenization, sampling)
└─ manifest.json + shard_*.rdrr (weight data only)
```

### Comparison

| Aspect | WASM (WebLLM) | JavaScript (DOPPLER) |
|--------|---------------|----------------------|
| CPU performance | ~2-3x faster | JS engine overhead |
| Debugging | Hard (compiled binary) | DevTools, breakpoints |
| Hot reload | Recompile model | Just refresh browser |
| Extensibility | C++ expertise | JS/TS expertise |
| Dynamic loading | Fixed at compile | Runtime flexibility |
| P2P integration | Awkward (binary blob) | Native (JS fetch/WebRTC) |

### Why JavaScript is Acceptable

**Bet: GPU fusion minimizes CPU work.**

If decode step is 26ms total:
- GPU compute: 25ms (96%)
- JS orchestration: 0.5ms (2%)
- Logits readback: 0.5ms (2%)

WASM would make the 0.5ms faster, but **who cares?** The bottleneck is GPU, not CPU.

> "For thin orchestration (dispatch commands, sample one token), JS overhead is ~0.5ms per decode step. This is 2% of a 26ms decode cycle."
>


### Why JavaScript is Advantageous

1. **Dynamic weight loading**: `fetch()` + `device.queue.writeBuffer()` - trivial in JS
2. **P2P integration**: WebRTC is a JS API, not available in WASM
3. **Debugging**: Chrome DevTools for the entire inference pipeline
4. **Rapid iteration**: No compilation step for model changes
5. **Ecosystem**: npm packages for tokenizers, UI, networking

---

## TVM Compilation vs Direct WGSL

### TVM Approach (WebLLM)

Apache TVM uses machine learning to auto-tune kernel configurations.

> "With an expressive code generator and an efficient search algorithm, we are able to generate kernels that are comparable to heavily hand-optimized ones."
>
> Source: [TVM Blog](https://tvm.apache.org/2018/10/03/auto-opt-all)

**Advantages:**
- Auto-tuning finds optimal tile sizes per device
- Cross-platform compilation
- Less manual optimization work

> "When compared against NCNN, a widely used hand-optimized kernel library that makes extensive use of NEON assembly instructions (with 13k lines of code for only 3x3 convolution layers), TVM outperforms it for all networks on Raspberry Pi 3B."
>
> Source: [TVM Mobile Optimization](https://tvm.apache.org/2018/01/16/opt-mali-gpu)

**Disadvantages:**
- Black box - harder to debug
- Some ops poorly optimized:

> "The scatter_nd op was reported to be almost 1000x slower than a naive hand-written CUDA implementation in one case."
>
> Source: [TVM Discussion](https://www.mail-archive.com/dev@tvm.apache.org/msg03451.html)

- Requires compilation step to add new models
- Model is monolithic unit (can't page experts)

### Direct WGSL Approach (DOPPLER)

**Advantages:**
- Full control over memory layout and access patterns
- Can implement cutting-edge algorithms directly (Flash Attention, fused MoE)
- No compiler dependency or black box
- Generic kernels work with any compatible model weights
- Dynamic expert paging (bind different weight buffers to same kernel)
- P2P-friendly (distribute data shards, not code)

**Disadvantages:**
- Significant engineering effort (~100KB of shader code)
- Must manually optimize for each GPU architecture
- Higher bug risk without compiler validation
- No auto-tuning (must implement separately)
- ~20% performance gap vs TVM auto-tuned kernels

Note: "Direct WGSL" means kernels are written directly without a compiler like TVM, not necessarily by hand. LLM coding agents can write WGSL, but there's no automated compilation/optimization pipeline.

### Industry Context

> "First-generation AI frameworks like TensorFlow and PyTorch 1.0 relied heavily on hand-written CUDA kernels, which couldn't scale to rapidly evolving AI workloads. TVM and XLA, as second-generation approaches, tackled this problem with automated compilation."
>
> Source: [Modular Blog](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)

DOPPLER takes a "direct kernel" approach for WebGPU - writing WGSL kernels without a compiler pipeline like TVM. This trades auto-tuning for **runtime flexibility**. The key distinction is:

| Approach | Unit of Distribution | Flexibility |
|----------|---------------------|-------------|
| TVM/WebLLM | Compiled model binary | None (fixed at compile) |
| DOPPLER | Weight shards + shared kernels | Full (paging, P2P, LoRA) |

---

## P2P and Evolution Potential

The JavaScript + runtime WGSL architecture unlocks P2P evolution of **dynamic components** - something impossible with pre-compiled approaches.

### The Key Insight: Static vs Dynamic

| Component | Size | P2P Value | Recommendation |
|-----------|------|-----------|----------------|
| **Weight shards** | 64MB each | Low - CDN handles fine | HuggingFace/CDN |
| **WGSL kernels** | ~5KB each | **High** - device-specific evolution | P2P swarm |
| **LoRA adapters** | 50-200MB | **High** - personality/domain tunes | P2P swarm |
| **Sampling strategies** | ~10KB | **High** - novel algorithms | P2P swarm |
| **Router weights** | ~1MB | **Medium** - learned MoE routing | P2P swarm |
| **Chat templates** | ~1KB | **Medium** - effective prompts | P2P swarm |

**Static weights** are best served by CDN (HuggingFace, Cloudflare). **Dynamic components** benefit from decentralized evolution.

### 1. Kernel Evolution (Primary P2P Value)

WebLLM's TVM-compiled kernels are frozen at build time. DOPPLER's WGSL kernels are plain text that can evolve:

```javascript
// User A discovers 2x faster attention on M3 Max
const kernel = await swarm.fetchKernel({
  name: 'attention_flash_v3',
  device: 'apple-m3-max',
  hash: 'sha256:abc123...'
});

// Benchmark locally, confirm improvement
const speedup = await benchmarkKernel(kernel, baseline);
if (speedup > 1.2) {
  swarm.endorse(kernel.hash);  // Signal to other M3 Max users
}
```

| Evolution Type | WebLLM | DOPPLER |
|----------------|--------|---------|
| Device-specific optimization | Impossible (one binary) | Natural (kernel per device class) |
| Community kernel improvements | Requires TVM recompile | Hot-swap at runtime |
| A/B testing kernel variants | Ship multiple binaries | Runtime benchmark + select |
| Rollback bad kernel | Redistribute binary | Revert to previous hash |

### 2. LoRA and Adapter Sharing

LoRA adapters are small (50-200MB) and high-value - perfect for P2P:

```javascript
// Fetch community LoRA for creative writing
const lora = await swarm.fetchAdapter({
  name: 'creative-writing-v2',
  baseModel: 'gemma-3-4b',
  hash: 'sha256:def456...'
});

// Apply without recompiling model
pipeline.applyLoRA(lora, alpha=0.7);
```

**WebLLM cannot do this** - LoRA must be fused at TVM compile time.

#### Current LoRA Ecosystem (Dec 2025)

No true P2P LoRA sharing network exists yet. Current ecosystem is centralized:

| Platform | Scale | Distribution |
|----------|-------|--------------|
| [HuggingFace](https://huggingface.co) | ~143,920 LoRAs (Oct 2024) | Centralized CDN |
| [Civitai](https://civitai.com) | Largest for image LoRAs | Centralized |
| [Tensor.Art](https://tensor.art) | Growing alternative | Centralized |

**Academic research:**
- **Dec-LoRA** ([arXiv:2501.15361](https://arxiv.org/abs/2501.15361), Jan 2025): Decentralized LoRA *training* - clients train locally, exchange updates with neighbors, no central server. 4-bit quantization reduces bandwidth. Privacy-preserving (data never leaves client).

**Edge runtimes:**
- **QVAC Fabric LLM** ([Tether.io](https://tether.io/news/tether-data-introduces-qvac-fabric-llm-the-edge-first-llm-inference-runtime-and-generalized-llm-lora-fine-tuning-framework-for-modern-ai-models-on-heterogeneous-gpus-smartphones-laptops-and-server/), Dec 2025): Open-source runtime for LoRA inference on phones/laptops. Apache 2.0 license. Not P2P sharing, but enables local execution.

**Gap:** IPFS/BitTorrent for LoRA distribution not found in production. Everyone uses HuggingFace CDN. This is DOPPLER's opportunity - first browser-native P2P LoRA sharing.

### 3. Sampling Strategy Sharing

Novel sampling algorithms discovered and shared:

```javascript
// Someone implements better speculative decoding
const sampler = await swarm.fetchSampler('speculative-tree-v2');

// Or mirostat for better coherence
const mirostat = await swarm.fetchSampler('mirostat-2.1');

pipeline.setSampler(sampler);
```

### 4. Collective Tuning Data

Swarm shares anonymized performance metrics:

```javascript
swarm.reportMetrics({
  device: 'apple-m2-pro',
  kernel: 'attention_flash_v2',
  tokensPerSecond: 42.5,
  expertHitRates: { expert_12: 0.23, expert_47: 0.18, ... }
});

// Aggregate helps everyone
const bestKernel = swarm.getBestKernel('attention', myDevice);
const hotExperts = swarm.getHotExperts('mixtral-8x7b');  // Pre-warm cache
```

---

### Weight Shard P2P (Secondary, With Caveats)

P2P distribution of weight shards IS architecturally possible but has significant challenges:

| Aspect | Benefit | Challenge |
|--------|---------|-----------|
| Cold start elimination | Nearby peers faster than CDN | Requires peer density, online peers |
| Bandwidth multiplication | N peers = Nx aggregate | WebRTC ~500KB/s limit per connection |
| Partial model serving | 600B across swarm | 200ms latency per expert fetch |
| Heterogeneous swarm | Peers contribute what they have | Coordination complexity, peer churn |

**Practical challenges:**
- NAT traversal fails 15-30% of connections
- Users must keep tab open to seed (why would they?)
- No browser P2P project has achieved critical mass
- CDNs (Cloudflare, HuggingFace) are already fast and free

**Recommendation:** Use CDN for weight shards. Reserve P2P for dynamic components where the evolution value outweighs coordination costs.

---

### 5. Swarm Intelligence

The combination enables emergent optimization:

```
┌─────────────────────────────────────────────────────────────┐
│                    DOPPLER Swarm                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Peer A (8GB VRAM)         Peer B (24GB VRAM)               │
│  ├─ Experts 0-49           ├─ Experts 0-255                 │
│  ├─ Uses attention_v2      ├─ Uses attention_flash          │
│  ├─ 25 tok/s               ├─ 45 tok/s                      │
│  └─ Shares: hit rates,     └─ Shares: hit rates,            │
│     latencies, kernel         latencies, kernel             │
│     perf metrics              perf metrics                  │
│                                                              │
│  Swarm Coordinator (any peer or dedicated)                  │
│  ├─ Aggregates metrics                                       │
│  ├─ Routes queries to optimal peer                          │
│  ├─ Suggests shard redistribution                           │
│  └─ Propagates best-performing kernels                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**This is impossible with compiled binaries.** WebLLM's WASM approach creates islands; DOPPLER's JS approach creates a mesh.

---

## DOPPLER Technical Differentiators

### 1. Multi-Tier Flash Attention in WGSL

DOPPLER implements **multi-tier Flash Attention** with automatic kernel selection based on device capabilities and model architecture:

| Tier | Kernel | headDim | Shared Memory | Use Case |
|------|--------|---------|---------------|----------|
| Large | `attention.wgsl` | <= 64 | 48KB | Standard models (Llama, Mistral) |
| Small | `attention_small.wgsl` | <= 256 | 8KB | Large heads (Gemma 3, custom) |
| Streaming | `attention_streaming.wgsl` | Any | None | Fallback for constrained devices |

```javascript
// Automatic tier selection in gpu/kernels/attention.js
const canLarge = headDim <= 64 && sharedLimit >= 49152;
const canSmall = headDim <= 256 && sharedLimit >= smallRequired;
tier = canLarge ? 'tiled_large' : canSmall ? 'tiled_small' : 'streaming';
```

**No other browser framework** implements Flash Attention directly in WGSL. WebLLM uses TVM-compiled attention kernels.

Features:
- Online softmax (numerically stable)
- Grouped Query Attention (GQA) support
- Causal masking with absolute position tracking
- Tiled computation to avoid full attention matrix materialization
- f16 KV cache support (`_f16kv` variants) for 2x memory savings
- headDim tiling for large head dimensions (Gemma 3 4B: headDim=256)

### 2. GPU-Native MoE Routing (Dec 2025)

Full mixture-of-experts execution on GPU with zero CPU readback:

```
inputBuffer (GPU)
    |
computeRouterLogitsGPU() -> logitsBuffer (GPU matmul)
    |
runSoftmaxTopK() -> indices, weights (GPU fused softmax+topk)
    |
_runExpertGPU() x numExperts -> expertOutputsBuffer (GPU FFN per expert)
    |
runScatterAdd() -> outputBuffer (GPU weighted combination)
    |
outputBuffer (GPU) <- stays on GPU, no readback
```

**Custom WGSL kernels:**

| Kernel | Purpose |
|--------|---------|
| `topk.wgsl` | Top-K selection with 3 variants (default, small k=2/n<=8, fused softmax+topk) |
| `scatter_add.wgsl` | Weighted scatter-add for combining expert outputs (vec4 + accumulate variants) |
| `moe_gather.wgsl` | Token gathering by expert (available, not used in current impl) |

**Implementation note:** Runs all experts for all tokens, then uses scatter-add to select top-k contributions. Simpler than gather-compute-scatter for typical decode batches.

### 3. Tiered Memory System

| Tier | Hardware | Max Model |
|------|----------|-----------|
| 1 | Unified Memory (Apple Silicon) | 60GB |
| 2 | Memory64 (discrete GPU) | 40GB MoE |
| 3 | Basic | 8GB small MoE |

**Validation needed:** 60GB claim is theoretical based on unified memory architecture. No real-world testing with models this large.

### 4. Native Bridge for Local Files

Shell script bridge (`bridge/native/doppler-bridge.sh`) enables mmap access to local model files, bypassing OPFS limits.

```javascript
// Load via Native Bridge (mmap)
const bridgeClient = await createBridgeClient();
const manifestBytes = await bridgeClient.read(manifestPath);
```

No other browser LLM framework offers native file access.

---

## WGSL Kernel Inventory

| Kernel | Lines | Purpose |
|--------|-------|---------|
| `attention.wgsl` | 340+ | Flash Attention (tiled, large, headDim<=64) |
| `attention_small.wgsl` | 200+ | Flash Attention (tiled, small, headDim<=256) |
| `attention_streaming.wgsl` | 100+ | Flash Attention (no shared mem fallback) |
| `attention_f16kv.wgsl` | 340+ | Attention with f16 KV cache (large) |
| `attention_small_f16kv.wgsl` | 200+ | Attention with f16 KV cache (small) |
| `attention_streaming_f16kv.wgsl` | 100+ | Attention with f16 KV cache (streaming) |
| `matmul_f16.wgsl` | 130+ | FP16 matrix multiplication |
| `matmul_f32.wgsl` | 85+ | FP32 matrix multiplication |
| `matmul_f16w_f32a.wgsl` | 120+ | Mixed precision (f16 weights, f32 activations) |
| `rmsnorm.wgsl` | 250+ | RMS normalization |
| `rope.wgsl` | 320+ | Rotary position embeddings |
| `softmax.wgsl` | 360+ | Softmax with online normalization |
| `silu.wgsl` | 210+ | SiLU activation (gated) |
| `topk.wgsl` | 230+ | Top-K selection (3 variants) |
| `scatter_add.wgsl` | 200+ | MoE output combination |
| `moe_gather.wgsl` | 220+ | Token gathering by expert |
| `dequant_shared.wgsl` | 200+ | Dequantization (shared memory) |
| `dequant_subgroup.wgsl` | 170+ | Dequantization (subgroup ops) |
| `dequant_f16_out.wgsl` | 150+ | Dequantization with f16 output |
| `cast_f32_to_f16.wgsl` | 40+ | Type casting for KV cache |
| `gather.wgsl` | 80+ | Embedding lookup |
| `residual.wgsl` | 65+ | Residual addition |

**Total:** ~100KB+ of custom WGSL shader code (direct, not TVM-compiled)

---

## Open Questions & Validation Needed

### Performance Benchmarks

- [ ] Tokens/sec vs WebLLM on same model (e.g., Llama 3 8B INT4)
- [ ] Tokens/sec vs WeInfer (claimed 3.76x over WebLLM)
- [ ] Prefill latency comparison
- [ ] Memory bandwidth utilization

### Large Model Support

- [ ] Actually run a 40GB+ model on unified memory Mac
- [ ] Verify MoE expert swapping works for Mixtral-class models
- [ ] Test Native Bridge mmap performance vs OPFS

### Browser Compatibility

- [ ] Safari buffer limit workarounds
- [ ] Firefox 141+ WebGPU testing
- [ ] Chrome on Android performance

### Kernel Correctness

- [ ] Flash Attention numerical accuracy vs reference
- [ ] MoE routing correctness with ground truth
- [ ] Quantization accuracy (INT4 vs FP16 vs FP32)

---

## Chrome Built-in AI (window.ai / Gemini Nano)

**Status (Dec 2025):** Available in Chrome Canary/Beta 127+, limited to desktop platforms.

> "Chrome's built-in AI (Gemini Nano) runs locally in the browser, offering zero-latency inference and complete privacy. The tradeoff: strict resource constraints — limited context windows (~2K-4K tokens), sequential execution to avoid crashes."
>
> Source: [Chrome Built-in AI Docs](https://developer.chrome.com/docs/ai/built-in)

**Available APIs:**
- Prompt API (web + extensions)
- Summarizer, Writer, Rewriter APIs
- Translator API (W3C WebML Working Group)
- Language Detector API

**System Requirements:**
- Platforms: Windows 10/11, macOS 13+, Linux (NOT Android, iOS, ChromeOS)
- Hardware: 22GB+ free disk, 4GB+ VRAM
- Model auto-downloads on first use

**Threat Assessment:**
- **Low threat to DOPPLER** - Limited to small models (Gemini Nano)
- 2-4K context window unsuitable for most LLM workloads
- No custom model support - locked to Google's models
- Useful for lightweight tasks (summarization, rewriting)

**Google I/O 2025:** Announced broader Gemini integration in Chrome desktop for AI Pro/Ultra subscribers.

**Track:** [developer.chrome.com/docs/ai](https://developer.chrome.com/docs/ai)

---

## ONNX Runtime Web: 4GB Limit Update

**Current State (Dec 2025):**

> "Currently, there is no way for ONNX Runtime Web to run models larger than 4GB."
>
> Source: [ONNX Runtime Docs](https://onnxruntime.ai/docs/tutorials/web/large-models.html), Dec 2025

**WASM64 Progress:**
- **WASM64 support now available** (build from source only)
- No pre-built packages published yet
- Source: [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)

**Planned Solutions:**
1. **WASM64** - 64-bit addressing removes 4GB cap (**in progress**)
2. **Direct GPU Weight Loading** - Bypass WASM entirely for weights

**Impact on DOPPLER:**
When WASM64 packages ship, Transformers.js (1.4M monthly users) gains large model support. Timeline unclear but likely 2025-2026.

**Track:** [GitHub Issue #13006](https://github.com/microsoft/onnxruntime/issues/13006)

---

## Distributed Inference Competitors

### Petals - BitTorrent-Style LLM Inference

Petals enables distributed inference across volunteer GPUs, running models up to 405B parameters.

> "Single-batch inference runs at up to 6 tokens/sec for Llama 2 (70B) and up to 4 tokens/sec for Falcon (180B) — enough for chatbots and interactive apps."
>
> Source: [petals.dev](https://petals.dev/)

**Key Features:**
- Splits model into blocks hosted on different servers
- Anyone can contribute GPU resources
- Web interface at [chat.petals.dev](https://chat.petals.dev) (WebSocket API)
- Supports Llama 3.1 (405B), Mixtral (8x22B), Falcon (40B+), BLOOM (176B)

**Relevance to DOPPLER:**
- Petals is server-side Python, not browser-based
- However, the swarm architecture is similar to DOPPLER's P2P proposal
- DOPPLER could potentially connect to Petals network for remote inference offload

### Federated Attention (FedAttn) - Research

> "FedAttn enables participants to perform local self-attention over their own token representations while periodically exchanging and aggregating Key-Value (KV) matrices."
>
> Source: [arXiv:2511.02647](https://arxiv.org/abs/2511.02647)

**Key Innovation:** Distributed attention without exposing raw data - only KV matrices exchanged.

**Relevance:** Academic research, not production-ready. But KV exchange pattern could inform DOPPLER's remote inference design.

### WebFLex - Browser P2P Federated Learning

> "WebFLex utilizes peer-to-peer interactions and secure weight exchanges utilizing browser-to-browser WebRTC, efficiently preventing the need for a main central server."
>
> Source: [ScienceDirect](https://www.sciencedirect.com/org/science/article/pii/S154622182400359X)

**Relevance:** Training-focused (federated learning), not inference. But demonstrates WebRTC P2P for ML weights in browser.

---

## P2P Model Distribution Landscape

### IPFS for ML Models

> "IPFS's cryptographic hashing preserves data integrity, with built-in versioning, making it easy to manage datasets and models efficiently."
>
> Source: [Preprints.org - Decentralizing AI with IPFS](https://www.preprints.org/manuscript/202411.0565/v1)

**Community efforts:**
- [pollinations/ipfs_model_hosting](https://github.com/pollinations/ipfs_model_hosting) - Hosting models on IPFS

**DOPPLER consideration:**
- IPFS could serve as alternative to WebTorrent for shard distribution
- Content-addressed hashing aligns with RDRR manifest design
- IPFS gateway fallback could improve cold start

### WebTorrent vs IPFS

Both are viable for P2P shard distribution:
- **WebTorrent:** Browser-native, mature WebRTC stack, tracker-based discovery
- **IPFS:** DHT-based discovery, content routing, larger ecosystem

DOPPLER's P2P proposal currently uses WebTorrent. IPFS could be a future alternative.

---

## Competitive Threat Timeline (Updated Dec 2025)

| Threat | Likelihood | Timeframe | Impact on DOPPLER |
|--------|------------|-----------|-------------------|
| WebLLM adopts buffer reuse/async | High | 2025-2026 | **High** - incumbent gains 3.76x edge |
| ONNX WASM64 packages | Medium | 2025-2026 | **High** - removes 4GB limit for Transformers.js |
| Chrome built-in AI expansion | Low | 2025-2026 | Low - locked to small Gemini Nano models |
| MediaPipe model expansion | High | Ongoing | Low - still Google-model focused |
| FlashInfer WebGPU port | Low | 2026+ | **High** - would match DOPPLER's attention perf |
| Petals browser client | Low | 2026+ | Medium - enables distributed inference |

Note: WeInfer itself is not a threat (stale since Feb 2025), but its techniques could be adopted by WebLLM.

### Defensive Priorities (Updated Dec 2025)

1. **CRITICAL:** Implement buffer reuse strategy (see WeInfer paper, not stale code)
2. **CRITICAL:** Implement async pipeline + deferred readback
3. **CRITICAL:** Command buffer batching (single submit per forward)
4. **Urgent:** Validate performance vs WebLLM baseline
5. **High:** Ship working MoE demo (Mixtral or similar)
6. **Medium:** Document Native Bridge advantages over OPFS
7. **Medium:** Ship P2P shard cache (tracked in feature-log)

### DOPPLER's Defensible Moats

| Differentiator | Threat Level | Notes |
|----------------|--------------|-------|
| Flash Attention in WGSL | Safe (2025) | No competitor has this in browser |
| Custom MoE routing | Contested | WebLLM supports MoE. Battle is on performance |
| 60GB unified memory | Untested | If validated, unique advantage |
| Native Bridge (mmap) | Unique | No competitor offers local file access |
| P2P shard cache | Planned | Swarm shard distribution reduces origin bandwidth and improves cold start |
| Custom WGSL (no compiler) | Double-edged | More control, but requires manual optimization |

---

## Appendix: Sources & Citations

### Primary Sources (with dates)

1. **WebLLM Paper** (Dec 2024)
   - URL: https://arxiv.org/abs/2412.15803
   - Claims: 80% native performance
   - Accessed: Dec 2025

2. **WeInfer Paper** (April 2025)
   - URL: https://dl.acm.org/doi/10.1145/3696410.3714553
   - OpenReview: https://openreview.net/forum?id=Qu2itILaoZ
   - Claims: 3.76x speedup over WebLLM v0.2.46
   - Conference: ACM Web Conference 2025

3. **WebGPU Buffer Bugs Analysis** (2024)
   - URL: https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca
   - Key insight: 35% user compatibility issues, 128-993MB buffer limits

4. **ONNX Runtime Large Models** (Dec 2025)
   - URL: https://onnxruntime.ai/docs/tutorials/web/large-models.html
   - GitHub Issue: https://github.com/microsoft/onnxruntime/issues/13006
   - Key insight: 4GB WASM hard limit, WASM64 planned

5. **Google MediaPipe 7B Blog** (2024)
   - URL: https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/
   - Key insight: Custom workarounds for 2GB ArrayBuffer limit

6. **Transformers.js v3 Blog** (Oct 2024)
   - URL: https://huggingface.co/blog/transformersjs-v3
   - Key insight: WebGPU support, 155 architectures, 1.4M users

7. **JSNation 2025 Talk** (2025)
   - URL: https://gitnation.com/contents/transformersjs-state-of-the-art-machine-learning-for-the-web
   - Key insight: Current Transformers.js scale and roadmap

8. **NVIDIA MoE Blog** (2025)
   - URL: https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/
   - Key insight: "Nearly all leading frontier models use MoE designs"

9. **Red Hat vLLM + DeepSeek** (Sept 2025)
   - URL: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-and-sparse-moe-models-vllm-llm-d
   - Key insight: Server-side MoE support in vLLM

10. **TVM Auto Optimization** (Oct 2018)
    - URL: https://tvm.apache.org/2018/10/03/auto-opt-all
    - Key insight: Comparable to hand-optimized kernels

11. **Modular on AI Compilers** (2024)
    - URL: https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers
    - Key insight: TVM/XLA as second-gen approach vs direct kernel authoring

12. **Dec-LoRA: Decentralized Low-Rank Fine-Tuning** (Jan 2025)
    - URL: https://arxiv.org/abs/2501.15361
    - Key insight: First decentralized LoRA training algorithm, no central server, 4-bit quantization

13. **QVAC Fabric LLM** (Dec 2025)
    - URL: https://tether.io/news/tether-data-introduces-qvac-fabric-llm-the-edge-first-llm-inference-runtime-and-generalized-llm-lora-fine-tuning-framework-for-modern-ai-models-on-heterogeneous-gpus-smartphones-laptops-and-server/
    - Key insight: Edge-first LoRA inference on phones/laptops, Apache 2.0

14. **Civitai LoRA Training Guide** (2025)
    - URL: https://civitai.com/articles/1716/opinionated-guide-to-all-lora-training-2025-update
    - Key insight: Largest LoRA sharing platform for image models

15. **Civitai Alternatives Analysis** (2025)
    - URL: https://neurocanvas.net/blog/civitai-alternatives-2025/
    - Key insight: HuggingFace has ~143,920 LoRAs (Oct 2024), no P2P distribution exists

### GitHub Repositories

| Repository | Stars | Last Checked |
|------------|-------|--------------|
| [WebLLM](https://github.com/mlc-ai/web-llm) | 16.9k | Dec 2025 |
| [Transformers.js](https://github.com/huggingface/transformers.js) | n/a | Dec 2025 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | n/a | Dec 2025 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | n/a | Dec 2025 |

### Additional Resources

- AI in Browser with WebGPU Guide: https://aicompetence.org/ai-in-browser-with-webgpu/
- WebLLM Model List: https://github.com/mlc-ai/web-llm/issues/683 (Dec 2025)
- TVM Mobile GPU Optimization: https://tvm.apache.org/2018/01/16/opt-mali-gpu (Jan 2018)
- MediaPipe Web Guide: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/web_js (Dec 2025)

---

*Last updated: December 30, 2025*

---

## Related Open Source Projects & Open Issues

### Major Related Projects

| Project | Description | Stars | Status |
|---------|-------------|-------|--------|
| [WebLLM](https://github.com/mlc-ai/web-llm) | High-performance in-browser LLM inference | 17k | Active |
| [Transformers.js](https://github.com/huggingface/transformers.js) | HuggingFace ML for the web | - | Active |
| [wgpu](https://github.com/gfx-rs/wgpu) | Rust cross-platform WebGPU API | - | Active |
| [Burn](https://github.com/tracel-ai/burn) | Rust deep learning with WGPU backend | - | Active |
| [Candle](https://github.com/huggingface/candle) | Minimalist Rust ML framework | - | WebGPU requested |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Microsoft's ONNX inference | - | Active |

---

### Unassigned/Open Issues by Project

#### WebLLM (mlc-ai/web-llm)

| Issue | Description | Status |
|-------|-------------|--------|
| [#711](https://github.com/mlc-ai/web-llm/issues/711) | WebLLM model provider for Vercel AI SDK | Unassigned |
| [#707](https://github.com/mlc-ai/web-llm/issues/707) | Roadmap: XGrammar, Phi-4, Gemma3 support | Open |
| [#718](https://github.com/mlc-ai/web-llm/issues/718) | Project activity status | Unassigned |

Full list: [Unassigned issues](https://github.com/mlc-ai/web-llm/issues?q=is%3Aissue+is%3Aopen+no%3Aassignee)

#### Transformers.js (huggingface)

| Issue | Description | Status |
|-------|-------------|--------|
| [#1469](https://github.com/huggingface/transformers.js/issues/1469) | WebGPU JSEP crashes on Gemma 3 (1b-it) | Dec 2025 |
| [#1416](https://github.com/huggingface/transformers.js/issues/1416) | Rotary interleaved attention not supported (MobileLLM) | Open |
| [#1425](https://github.com/huggingface/transformers.js/issues/1425) | text-to-speech-webgpu build fails | Open |
| [#1380](https://github.com/huggingface/transformers.js/issues/1380) | WebGPU crash on translation pipeline | Open |
| [#1317](https://github.com/huggingface/transformers.js/issues/1317) | WebGPU broken with q8 decoders | Open |
| [#1289](https://github.com/huggingface/transformers.js/issues/1289) | WebGPU not used in webgpu-chat | Open |
| [#1205](https://github.com/huggingface/transformers.js/issues/1205) | WebGPU crash on Android Chrome | Open |

#### ONNX Runtime Web (microsoft)

| Issue | Description | Status |
|-------|-------------|--------|
| [#26295](https://github.com/microsoft/onnxruntime/issues/26295) | WebGPU from Python documentation | Oct 2025 |
| [#26216](https://github.com/microsoft/onnxruntime/issues/26216) | WebGPU EP in onnxruntime-node | Oct 2025 |
| [#26107](https://github.com/microsoft/onnxruntime/issues/26107) | Custom WebGPU device not used | Sep 2025 |
| [#24442](https://github.com/microsoft/onnxruntime/issues/24442) | Incorrect predictions on Intel GPUs | Open |
| [#20876](https://github.com/microsoft/onnxruntime/issues/20876) | WebGPU unavailable in Service Worker | Open |

#### wgpu (gfx-rs)

| Issue | Description | Status |
|-------|-------------|--------|
| [#7197](https://github.com/gfx-rs/wgpu/issues/7197) | Mesh Shaders tracking issue | Open |
| [#8010](https://github.com/gfx-rs/wgpu/issues/8010) | Learning resources feedback | Open |

Full list: [wgpu issues](https://github.com/gfx-rs/wgpu/issues)

#### Candle (huggingface)

| Issue | Description | Status |
|-------|-------------|--------|
| [#344](https://github.com/huggingface/candle/issues/344) | **WebGPU support** (31+ 👍, highly requested) | Open |
| [#346](https://github.com/huggingface/candle/issues/346) | AMD hardware support | Open |

Full list: [Candle issues](https://github.com/huggingface/candle/issues)

#### Burn (tracel-ai)

- WebGPU backend via `burn-wgpu` crate
- Issues: [burn issues](https://github.com/tracel-ai/burn/issues)

---

### High-Impact Contribution Opportunities

| Issue | Project | Why It Matters | DOPPLER Relevance |
|-------|---------|----------------|-------------------|
| Gemma 3 WebGPU crash (#1469) | Transformers.js | Directly related to Doppler's Gemma 3 support | **High** - shared debugging insights |
| WebGPU support (#344) | Candle | 31+ votes, would enable Rust→WASM→WebGPU ML | Medium - potential future backend |
| MobileLLM rotary attention (#1416) | Transformers.js | Missing kernel implementation | Medium - DOPPLER has custom RoPE |
| WebGPU in Service Worker (#20876) | ONNX Runtime | Enables extension-based inference | Medium - same browser constraints |
| Custom GPUDevice (#26107) | ONNX Runtime | Needed for shared device scenarios | Low - different architecture |

---

### Cross-Project Architecture Comparison

| Aspect | Gamma (PyTorch) | Doppler (WGSL) | WebLLM (TVM) | Transformers.js (ONNX) |
|--------|-----------------|----------------|--------------|------------------------|
| Runtime | Native Python | Browser WebGPU | Browser WebGPU | Browser WebGPU/WASM |
| Acceleration | CUDA/MPS | WebGPU shaders | TVM-compiled | ONNX Runtime |
| Performance | ~100% native | ~80% native | ~80% native | ~60-80% native |
| Model format | HuggingFace/GGUF | RDRR (64MB shards) | TVM compiled | ONNX |
| Max model | Unlimited | 60GB (theoretical) | ~31GB VRAM | **4GB hard limit** |
| MoE support | Yes | Yes (custom) | Yes (Mixtral) | No |
| Custom kernels | No | **Yes (WGSL)** | No (TVM) | No (ONNX ops) |
| RSI potential | No | **Yes** | No | No |

### WGSL vs PyTorch Design Philosophy

DOPPLER accepts ~20% kernel performance gap vs compiled approaches because it enables:
- **Expert paging** (90GB MoE on 8GB VRAM)
- **P2P shard distribution**
- **LoRA hot-swap** at runtime
- **Reploid can evolve kernels** during execution

The WGSL kernels in `gpu/kernels/` mirror PyTorch ops but are hand-written for WebGPU's compute model with entry point variants for different batch sizes (GEMV vs GEMM, decode vs prefill).

---

### Ecosystem Gaps DOPPLER Could Fill

| Gap | Current State | DOPPLER Opportunity |
|-----|---------------|---------------------|
| Browser MoE beyond WebLLM | Only WebLLM has Mixtral | Custom expert paging, larger models |
| Kernel hot-swap | No one does this | RSI-driven kernel evolution |
| P2P model distribution | No browser P2P exists | WebTorrent/IPFS shard sharing |
| LoRA runtime loading | WebLLM requires recompile | Dynamic adapter hot-swap |
| >4GB in ONNX ecosystem | Hard WASM limit | DOPPLER has no WASM dependency |
| Flash Attention in browser | Only DOPPLER | Validated differentiator |

---

## Additional Projects & Emerging Standards

### Hand-Tuned WebGPU Inference

| Project | Description | Status | Notes |
|---------|-------------|--------|-------|
| [token-hawk](https://github.com/kayvr/token-hawk) | Hand-written LLaMA WebGPU inference | Active | 37 tok/s on 4090 (7B-f16), GGML format only |
| [Ratchet](https://github.com/huggingface/ratchet) | HuggingFace cross-platform browser ML | Active | Rust/WASM, Whisper + Phi support, GGUF native |
| [whisper-web](https://github.com/xenova/whisper-web) | Real-time speech recognition in browser | Active | Uses Transformers.js + ONNX Runtime |
| [browser-llm-webgpu](https://github.com/hannes-sistemica/browser-llm-webgpu) | Reasoning model PoC in browser | PoC | WebGPU acceleration demo |

### token-hawk Details

> "TokenHawk is fast. On a 4090 using 7B-f16, TokenHawk clocks in at 37 tk/s with room for improvement."

- **Limitations**: Only 7B models (VRAM constraint), 512 token context tested
- **Dependencies**: Google Dawn (CLI), no deps for web
- **Relevance to DOPPLER**: Similar hand-tuned WGSL approach, validates direct kernel strategy

Source: [token-hawk GitHub](https://github.com/kayvr/token-hawk)

### Ratchet Details

> "A toolkit for developers to integrate performant AI functionality into existing production applications."

**Key Features:**
- Single model implementation for both full precision and quantized
- Transparent GGUF support - pull from HuggingFace, parse on-the-fly, transcode to WebGPU format
- IndexedDB caching (no ONNX conversion needed)
- TensorFlow.js-style memory allocator

**Challenge noted:** "WebGPU hasn't proliferated as fast and as widely as expected. Need fast WASM SIMD backend for compatibility."

Source: [Ratchet GitHub](https://github.com/huggingface/ratchet), [RFC: Ratchet V1](https://github.com/huggingface/ratchet/discussions/187)

---

### WebNN - Emerging Browser ML Standard

| Aspect | Current State (Dec 2025) |
|--------|--------------------------|
| Status | Preview - NOT production ready |
| Chrome | Available with flags |
| Edge | Available with flags |
| Safari | Not available |
| Firefox | Not available |

**Major 2025 Changes:**
- **May 2025**: DirectML deprecated at Microsoft Build
- WebNN now uses Windows ML → OpenVINO for hardware acceleration
- Backend order: ONNX Runtime → DirectML → TFLite (Windows), Core ML → TFLite (Apple)

**Key Issues (from Nov 2025 W3C meeting):**

| Issue | Description | Impact |
|-------|-------------|--------|
| [#901](https://github.com/webmachinelearning/webnn/issues/901) | Separate graph building from weight loading | 3x peak memory reduction |
| [#883](https://github.com/webmachinelearning/webnn/issues/883) | Dynamic shape support (symbolic sizes) | Flexible input dimensions |
| Performance | GroupQueryAttention = 24 WebNN ops | SLM needs macro ops |
| Memory | Can't use all available memory during graph building | Efficiency concern |

**Requirements:** Windows 11 24H2+, `kWebNNOnnxRuntime` flag

Sources: [W3C WebNN Spec](https://www.w3.org/TR/webnn/), [WebNN Overview](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview), [W3C Meeting Nov 2025](https://www.w3.org/2025/11/09-webmachinelearning-minutes.html)

---

### FlashAttention Implementations

| Project | Implementation | Status |
|---------|----------------|--------|
| **DOPPLER** | Custom WGSL (tiled, multi-tier) | Production |
| **ONNX Runtime** | WebGPU/WGSL ([PR #24400](https://github.com/microsoft/onnxruntime/pull/24400)) | Active |
| **WebLLM** | TVM-compiled | Production |

**ONNX Runtime FlashAttention (April 2025):** Fixed Unicode characters in WGSL comments causing Windows logging failures.

---

### Apache TVM WebGPU Status

| Item | Status | Notes |
|------|--------|-------|
| WebGPU Runtime | Stable | JavaScript-first, async API |
| FP16 Support | [Tracking #14905](https://github.com/apache/tvm/issues/14905) | WGSL codegen needed |
| Subgroup Shuffle | [Draft PR #17699](https://github.com/apache/tvm/pull/17699) | Warp-level primitives |

Source: [TVM GitHub](https://github.com/apache/tvm)

---

### whisper.cpp / Whisper WebGPU

| Variant | Runtime | Notes |
|---------|---------|-------|
| [whisper.cpp WASM](https://ggml.ai/whisper.cpp/) | WASM SIMD | Official demo, CPU-only |
| [Whisper WebGPU](https://huggingface.co/spaces/Xenova/realtime-whisper-webgpu) | WebGPU + ONNX | Real-time, 100 languages, ~200MB model |
| [whisper-web](https://github.com/xenova/whisper-web) | Transformers.js | Chrome-only WebGPU branch |

**Whisper WebGPU features:**
- Real-time in-browser processing
- 100 language transcription/translation
- Offline after initial load
- Uses Transformers.js + ONNX Runtime Web

Sources: [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [whisper-web](https://github.com/xenova/whisper-web)

---

### llama.cpp WebGPU Backend

The llama.cpp project has experimental WebGPU support via Dawn:

> "WebGPU allows cross-platform access to the GPU from supported browsers. They utilize Emscripten to compile ggml's WebGPU backend to WebAssembly."

**Current limitations:**
- Requires Dawn library locally
- Emscripten doesn't officially support WebGPU bindings yet
- Uses Dawn's `emdawnwebgpu` bindings

Source: [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

---

### WebGPU Browser Implementation Issues

| Browser | Issues | Notes |
|---------|--------|-------|
| **Chrome/Edge** | Multi-GPU limitations, power management bugs | WebGPU since v113 (April 2023) |
| **Firefox** | 90% spec compliance, crashes on invalid shaders | 3 full-time devs only, v141+ |
| **Safari** | Buffer limits (256MB-993MB), shipped June 2025 | Safari 26 |

**Shader Compilation Pain Points:**
- Complex LLM kernels take seconds to compile on first use
- Mobile devices can timeout entirely
- Chrome 141: New Tint IR backend = **7x speed improvement** for shader compilation

**Critical Issues:**
- 35% of users affected by WebGPU compatibility issues
- Buffer size limits prevent large model deployment
- "GpuProcessHost: The GPU process died due to out of memory" - no recovery mechanism

Sources: [WebGPU Bugs Article](https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca), [Chrome 141 WebGPU](https://developer.chrome.com/blog/new-in-webgpu-141)

---

### Additional MediaPipe Issues

| Issue | Description | Status |
|-------|-------------|--------|
| [#5974](https://github.com/google-ai-edge/mediapipe/issues/5974) | GPU implementation not in repo | May 2025, Unresolved |
| [#6093](https://github.com/google-ai-edge/mediapipe/issues/6093) | Invalid TFLite Flatbuffer (gemma3-1b-it-q4) | Sep 2025 |
| [#6100](https://github.com/google-ai-edge/mediapipe/issues/6100) | Can't measure WASM memory usage | Sep 2025 |
| [#5468](https://github.com/google-ai-edge/mediapipe/issues/5468) | No pre-download benchmark for device capability | Feature request |

Source: [MediaPipe Issues](https://github.com/google-ai-edge/mediapipe/issues)

---

### Competitive Positioning Summary

| Approach | Projects | Tradeoff |
|----------|----------|----------|
| **TVM-compiled** | WebLLM | Auto-tuned, frozen at compile |
| **ONNX Runtime** | Transformers.js | Broad model support, 4GB limit |
| **Hand-tuned WGSL** | DOPPLER, token-hawk | Maximum control, manual optimization |
| **Rust + WASM** | Ratchet, Candle | Type safety, WASM fallback |
| **Google-native** | MediaPipe | Gemma-focused, stable |

**DOPPLER's unique position:** Only project combining:
1. Hand-tuned WGSL kernels (like token-hawk)
2. MoE support (like WebLLM)
3. RSI/kernel evolution capability (unique)
4. P2P distribution architecture (planned, unique)

---

*Additional findings added: December 30, 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## GEPA Synergy

**Status:** Study Document
**Context:** How Reploid/Doppler can integrate and extend GEPA's prompt evolution approach

---

## TL;DR

GEPA provides the **learning algorithm** (evolutionary prompt mutation).
Reploid provides the **substrate infrastructure** (slots, rollback, safety).
Doppler provides the **compute layer** (local inference, LoRA swapping).

Together, they form a complete browser-native RSI platform. But Reploid/Doppler can also **go beyond GEPA** by evolving things GEPA cannot touch: executable code, kernel configurations, and runtime adapters.

---

## The Two Types of Merging

### Composition (Reploid) vs. Crossover (GEPA)

| Aspect | Reploid (Manual Composition) | GEPA (Evolutionary Crossover) |
|--------|------------------------------|-------------------------------|
| **Who decides** | Human operator | Automated evolution |
| **Logic** | "I think the agent needs these skills" | "Agent A excels at X, Agent B at Y, combine" |
| **Goal** | Modularity, organization | Genetic improvement, discovery |
| **Finds what** | Combinations you plan | Combinations you wouldn't think of |

**Reploid's PersonaManager:** Merges static lesson blocks (e.g., `Build a Tool` + `Analyze Reflections`)

**GEPA's System-Aware Merge:** Blindly combines Module 1 from Agent A with Module 2 from Agent B to create a super-agent

### Output Merging vs. Instruction Merging

| Aspect | Reploid (Arena/Consensus) | GEPA (Prompt Mutation) |
|--------|---------------------------|------------------------|
| **Merges** | Outputs from multiple models | Instructions themselves |
| **Effect** | More reliable *now* (ensemble) | Smarter *forever* (learning) |
| **Type** | Runtime decision | Evolutionary improvement |

---

## What Reploid/Doppler Can Evolve

### The Full RSI Surface

| Layer | What | Reploid | Doppler | GEPA Can Touch |
|-------|------|---------|---------|----------------|
| **L0** | System prompts | PersonaManager lessons | - | Yes (core GEPA) |
| **L1** | Tools | CreateTool, tool library | - | Partially (tool prompts) |
| **L2** | Meta-tools | Tool-writer, reflection loop | - | No (executable code) |
| **L3** | Substrate | agent-loop.js, core modules | - | No (executable code) |
| **K1** | Kernel configs | - | optimizations.kernelPath | Yes (explicit path) |
| **K2** | LoRA adapters | - | Runtime LoRA swap | No (weight deltas) |
| **K3** | Kernel variants | - | WGSL shader selection | No (GPU code) |
| **K4** | Expert routing | - | MoE router weights | No (learned routing) |

**Key insight:** GEPA only evolves natural language (prompts). Reploid/Doppler can evolve **executable artifacts** that GEPA cannot.

---

## Synergy: GEPA + Reploid Infrastructure

### What GEPA Needs (That Reploid Has)

| GEPA Requirement | Reploid Implementation |
|------------------|------------------------|
| Modular prompt slots | PersonaManager lessons |
| Execution traces | EventBus audit logs |
| Rollback mechanism | Genesis Snapshots |
| Diversity preservation | Arena consensus |
| Validation harness | Verification Worker |

### The Integration Flow

```
User Goal
    |
    v
Reploid Agent Loop (orchestration)
    |
    v
Doppler Inference (local LLM)
    |
    v
Tool Execution in VFS Sandbox
    |
    v
Execution Trace Capture (EventBus) <-- GEPA hooks in here
    |
    v
GEPA Reflection (diagnose failure)
    |
    v
Prompt Mutation Proposal
    |
    v
Arena Verification (multi-model consensus)
    |
    v
Genesis Checkpoint --> Apply --> Validate
    |
    v
Pareto Pool Update (or Rollback)
```

### What's Missing for Full GEPA Integration

1. **Structured reflection function ($\mu_f$):** EventBus captures traces but doesn't algorithmically process them
2. **Pareto pool tracking:** Genesis tracks snapshots, not evolutionary lineage
3. **Crossover operator:** PersonaManager allows composition but not genetic recombination

---

## "One-Upping" GEPA: Beyond Prompt Evolution

### 1. Executable Code Evolution (L1-L3)

GEPA evolves prompts. Reploid can evolve **actual JavaScript**.

```javascript
// GEPA: Evolve the PROMPT for a tool
"When searching, use multi-hop retrieval..."

// Reploid L1: Evolve the TOOL CODE itself
tools.SearchTool = function(query) {
  // This code can be modified by the agent
  return multiHopRetrieval(query, { hops: 3 });
};

// Reploid L2: Evolve the TOOL-WRITER that creates tools
// The meta-tool learns better patterns for tool construction
```

**Advantage:** Code evolution is more powerful than prompt evolution. A tool that implements binary search will always outperform a prompt that describes binary search.

**Safety trade-off:** This is why Reploid has 8 safety layers. Code evolution is dangerous.

### 2. Kernel Configuration Evolution (Doppler K1)

Doppler's kernel path overrides in RDRR manifests are evolvable:

```json
{
  "optimizations": {
    "kernelPath": "gemma2-q4k-fused-f16a"
  }
}
```

**Evolution loop:**
1. Run benchmark with configuration A
2. Reflect on profiler output (GPU timing, memory bandwidth)
3. Mutate configuration (try different workgroup sizes)
4. Validate (benchmark again)
5. Keep if Pareto-optimal for this device class

**GEPA quote (directly applicable):**
> "On NPU kernel generation, GPT-4o Baseline achieved 4.25% vector utilization. GEPA-optimized GPT-4o achieved 30.52% vector utilization."

Apply this to WGSL kernels: evolve the kernel path choice, not the prompts.

### 3. LoRA Adapter Evolution (Doppler K2)

Doppler supports runtime LoRA swapping. This enables **LoRA lineage tracking**:

```javascript
// LoRA Pareto pool
const loraPool = {
  'creative-v12': { wins: ['story', 'poetry'], avgScore: 0.82 },
  'technical-v8': { wins: ['code', 'docs'], avgScore: 0.79 },
  'hybrid-v3': { wins: ['mixed'], avgScore: 0.75 }
};

// Select based on task type
const lora = selectParetoBest(loraPool, taskType);
pipeline.applyLoRA(lora);
```

**Beyond GEPA:** LoRA is a weight delta. GEPA explicitly cannot modify weights. Doppler can swap LoRAs at runtime without recompilation.

### 4. P2P Kernel Evolution (Doppler Vision)

P2P kernel swarm concept:

```javascript
// User A discovers 2x faster attention on M3 Max
const kernel = await swarm.fetchKernel({
  name: 'attention_flash_v3',
  device: 'apple-m3-max',
  hash: 'sha256:abc123...'
});

// Benchmark locally, confirm improvement
const speedup = await benchmarkKernel(kernel, baseline);
if (speedup > 1.2) {
  swarm.endorse(kernel.hash);  // Propagate to other M3 users
}
```

**This is GEPA at the kernel level:**
- Reflect: Benchmark output
- Mutate: Swap kernel variant
- Evaluate: Confirm speedup
- Pareto: Endorse if best for this device class

**Advantage over GEPA:** Kernels are GPU code, not prompts. Evolution operates on a different substrate.

---

## Concrete Integration Opportunities

### 1. GEPA-Style Prompt Slot Evolution

Add to PersonaManager:

```javascript
class GEPAPersonaManager extends PersonaManager {
  constructor() {
    this.paretoPool = new Map(); // slot -> version -> { prompt, wins }
  }

  async evolveSlot(slotName, executionTrace, failureMode) {
    // GEPA reflection function
    const currentPrompt = this.getSlot(slotName);
    const mutatedPrompt = await this.reflect(currentPrompt, executionTrace, failureMode);

    // Validate mutation
    const score = await this.validate(mutatedPrompt, slotName);

    // Pareto update
    this.updateParetoPool(slotName, mutatedPrompt, score);
  }

  crossover(slotNameA, versionA, slotNameB, versionB) {
    // GEPA system-aware merge
    return this.merge(
      this.paretoPool.get(slotNameA).get(versionA),
      this.paretoPool.get(slotNameB).get(versionB)
    );
  }
}
```

### 2. Execution Trace Reflection Hook

Add to EventBus:

```javascript
// In EventBus subscriber
eventBus.subscribe('tool:complete', async (event) => {
  const trace = {
    tool: event.toolName,
    input: event.input,
    output: event.output,
    error: event.error,
    duration: event.duration
  };

  // Feed to GEPA reflection
  if (event.error || event.quality < threshold) {
    await gepa.reflectOnFailure(trace);
  }
});
```

### 3. Doppler Kernel Config Evolution

Add to benchmark harness:

```javascript
class KernelEvolver {
  constructor() {
    this.configPool = new Map(); // device -> config -> performance
  }

  async evolve(deviceClass) {
    const baseConfig = this.getBestConfig(deviceClass);

    // Mutation: try different workgroup sizes
    const mutations = this.generateMutations(baseConfig);

    for (const config of mutations) {
      const perf = await benchmark(config);
      this.updatePareto(deviceClass, config, perf);
    }
  }

  generateMutations(config) {
    // Try workgroup size variations
    const sizes = [[64, 1, 1], [128, 1, 1], [64, 4, 1], [32, 8, 1]];
    return sizes.map(wg => ({ ...config, workgroupSize: wg }));
  }
}
```

---

## What Reploid/Doppler Has That GEPA Lacks

### 1. Containment (Safety-First RSI)

GEPA assumes a controlled environment. Reploid assumes **mutations might be dangerous**:

| Safety Layer | Purpose | GEPA Equivalent |
|--------------|---------|-----------------|
| VFS Sandbox | No host file access | None |
| Genesis Snapshots | Immutable rollback | Discard mutation |
| Arena Consensus | Multi-model validation | Single validator |
| HITL Queue | Human approval | None |
| Circuit Breakers | Halt on anomaly | None |

### 2. Multi-Substrate Evolution

GEPA evolves one thing: prompts.

Reploid/Doppler can evolve **multiple substrates simultaneously**:
- Prompts (PersonaManager)
- Tools (CreateTool)
- Code (L2-L3 RSI)
- Kernels (optimizations.kernelPath)
- LoRAs (weight deltas)
- Router weights (MoE learned routing)

**Emergent behavior:** A mutation in tool code might enable a simpler prompt. Cross-substrate optimization is impossible in pure GEPA.

### 3. Local-First Compute

GEPA requires API calls to LLMs. Doppler runs **entirely in-browser**:

| Aspect | GEPA | Doppler + Reploid |
|--------|------|-------------------|
| Inference | Cloud API | Local WebGPU |
| Rollout cost | $ per token | Free (local) |
| Privacy | Data leaves device | Data stays local |
| Latency | Network bound | GPU bound |

**Sample efficiency matters more when rollouts are free.** GEPA's 35x efficiency over RL is impressive when rollouts cost money. With Doppler, you can run thousands of rollouts locally.

### 4. Weight-Space Access (LoRA)

GEPA explicitly rejects weight modification:
> "many downstream LLM applications... simply cannot finetune the weights of the largest or best-performing LLMs."

Doppler can swap LoRAs at runtime. This is **weight-space evolution without training**:
- Swap in creative-LoRA for storytelling
- Swap in technical-LoRA for coding
- Track which LoRA wins on which task type
- Build a Pareto pool of adapters

---

## Implementation Roadmap

### Phase 1: Trace Capture (Foundation)
- [ ] Structured execution trace format in EventBus
- [ ] Trace persistence in VFS for offline analysis
- [ ] Failure categorization (tool error, logic error, timeout)

### Phase 2: Reflection Function (GEPA Core)
- [ ] Implement $\mu_f$ reflection on traces
- [ ] Prompt mutation proposals
- [ ] Integration with PersonaManager slots

### Phase 3: Pareto Tracking (Diversity)
- [ ] Slot version lineage in Genesis
- [ ] Task-specific wins tracking
- [ ] Crossover operator for slot combination

### Phase 4: Multi-Substrate Evolution (Beyond GEPA)
- [ ] Kernel config evolution with benchmark feedback
- [ ] LoRA selection based on task type
- [ ] Tool code evolution (L1-L2)

### Phase 5: P2P Evolution (Swarm)
- [ ] Kernel variant sharing across devices
- [ ] LoRA Pareto pool distribution
- [ ] Collective performance metrics

---

## Key Quotes

### GEPA on Why Language > Gradients
> "When an agent fails a multi-step task because of a subtle logic error in step 3, a scalar reward of `0` tells the model *that* it failed, but not *why*."

### GEPA on Emergent Knowledge
> "GEPA effectively 'learned' the algorithm for multi-hop retrieval and wrote it down in English."

### GEPA on Sample Efficiency
> "GEPA attains optimal test set performance... with only 678 (35x fewer) rollouts."

### GEPA on Pareto Diversity
> "A candidate is kept in the pool if it is the best performing candidate for at least one specific task instance."

---

## Summary

| Dimension | GEPA | Reploid | Doppler | Combined |
|-----------|------|---------|---------|----------|
| **Evolves** | Prompts | Prompts + Code | Configs + LoRAs | Everything |
| **Safety** | Validation | Containment | N/A | Containment + Validation |
| **Compute** | Cloud API | N/A | Local WebGPU | Local-first |
| **Diversity** | Pareto pool | Arena consensus | Benchmark Pareto | Multi-level Pareto |
| **Rollback** | Discard | Genesis | N/A | Genesis + Version tracking |

**The thesis:** GEPA proves language is a sufficient optimization medium for prompts. Reploid/Doppler extend this to code, kernels, and weights—substrates GEPA cannot touch—while providing the safety infrastructure GEPA assumes but doesn't implement.

---

*Last updated: December 2025*
