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

We achieve this by inverting the standard stack. Instead of bringing the data to a centralized model, we mount a **Distributed Mixture-of-Experts (MoE)** directly to the browser runtime via DOPPLER. The P2P mesh becomes an infinite-capacity cache tier for model weights. KV cache is still GPU-resident today; paged KV layouts are future work (see ARCHITECTURE.md).

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
| GPU VRAM | 8-24GB | <1ms | Active experts, KV cache (paged KV is future) |
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
| 90GB MoE on 8GB VRAM | Expert paging | **No** | **Planned** |
| P2P kernel evolution | Plain-text WGSL | **No** | **Planned** |
| LoRA without recompile | Generic kernels | **No** | **Yes** |
| Device-specific optimization | Runtime kernel swap | **No** | **Planned** |
| Speculative decoding | Multi-model coordination | Awkward | **Native** |

---

## Why Not TVM? (Detailed)

WebLLM uses Apache TVM to pre-compile model-specific WGSL kernels with auto-tuning:

| Aspect | TVM/WebLLM | DOPPLER |
|--------|------------|---------|
| Kernel performance | ~80% native (auto-tuned) | ~60-70% native (manual) |
| New model support | Requires offline compilation | Runtime-compatible |
| Dynamic sharding | Not possible (fixed in binary) | Planned (expert paging) |
| P2P distribution | Distribute compiled .wasm | Planned (RDRR shards over P2P) |
| Model evolution | Recompile entire model | Planned (hotswap manifests) |
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

DOPPLER (planned): Generic FFN kernel + weight buffer binding
        ├─ Load expert_2 from OPFS → bind → run kernel
        ├─ Evict expert_2 (release buffer)
        ├─ Load expert_7 from peer → bind → same kernel
        └─ Run 90GB model on 8GB VRAM via paging (weights; KV cache remains VRAM-resident today)
```

### P2P Evolution (Dynamic Components)

The real P2P value is **evolving dynamic components**, not distributing static weights:

```
WebLLM: Kernels frozen in compiled binary
        └─ Everyone uses same TVM-generated kernels forever

DOPPLER (planned): Kernels are plain text, evolve across swarm
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

**Task tracking:** See this roadmap and project issues for the current task list.

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
| Internal postmortems (private wrapper repo) | Incident summaries |

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


## Technical Deep-Dives

For technical implementation details, see:
- `ARCHITECTURE.md` for subsystem breakdown and pipeline flow
- `CONFIG.md` for kernel path definitions and tuning knobs
- `FORMATS.md` for RDRR manifest fields and tensor layout

---

## Status Overview

For current operational status, test results, and recent fixes, see:
- Internal postmortems (private wrapper repo) for issue history
- [Architecture](ARCHITECTURE.md) for system design


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

Use the browser conversion UI in `/demo/` to convert to RDRR. For programmatic
conversion, import `src/browser/browser-converter.js` in the browser and write
to OPFS.


## Detailed Specs
- [Browser Training](design/BROWSER_TRAINING.md)
- [FunctionGemma Refactor](design/FUNCTION_GEMMA.md)
- [GEPA Synergy](design/GEPA_SYNERGY.md)
