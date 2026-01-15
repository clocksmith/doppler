# DOPPLER: Competitive Landscape & Technical Context

**DOPPLER** (Distributed Object Parallel Processing Layer Executing REPLOID) is a browser-native LLM inference engine. It is part of the REPLOID system (Recursive Evolution Protocol Loop Orchestrating Inference DOPPLER), with model distribution handled by RDRR (Recursive DOPPLER Runtime Registry).
See also: [Glossary](../GLOSSARY.md)

## TL;DR

DOPPLER is a browser-native LLM inference engine using custom WebGPU (WGSL) kernels (no compiler like TVM). Key differentiators:

1. **Flash Attention in WGSL** - No other browser framework implements tiled Flash Attention directly in WGSL
2. **Custom MoE routing** - Direct WGSL scatter-add (vs TVM-generated gather) for lower VRAM overhead
3. **60GB model support (theoretical)** - Tiered memory system for unified memory architectures
4. **Native Bridge** - mmap access to local files, bypassing OPFS limits

**Caveat:** Performance benchmarks pending. WebLLM supports MoE (Mixtral) via TVM. DOPPLER must prove better performance.
See: `docs/design/BENCHMARK_HARNESS.md` and `docs/design/KERNEL_TESTING.md`.

---

## Roadmap and Metrics

This document focuses on competitor context and technical constraints.

Implementation work, task tracking, and priorities live in the feature-log system
(`feature-log/doppler/*.jsonl`).

Benchmark and testing specs:

- `docs/design/BENCHMARK_HARNESS.md` (pipeline and system benchmarks, result schema)
- `docs/design/KERNEL_TESTING.md` (kernel and segment tests, how to interpret correctness)
- `docs/plans/TARGET_MODELS.md` (benchmark priority list)

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
> Source: [DOPPLER VISION.md](../VISION.md)

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
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes and the OPFS purge helper.
