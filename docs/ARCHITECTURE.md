# DOPPLER Architecture

**DOPPLER** (Distributed Object Parallel Processing Layer Executing REPLOID) is a WebGPU-native LLM inference engine for browser environments. It is part of the REPLOID system (Recursive Evolution Protocol Loop Orchestrating Inference DOPPLER).

See also: [Glossary](GLOSSARY.md)

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Application                             │
│                    (doppler-provider.ts API)                        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Inference Pipeline                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Tokenizer  │  │  KV Cache   │  │ MoE Router  │  │ Speculative│ │
│  │             │  │ (GPU/CPU)   │  │ (optional)  │  │  Decoder   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐
│   GPU Subsystem   │  │   Loader          │  │   Storage             │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────────┐  │
│  │   Device    │  │  │  │ DOPPLER     │  │  │  │  RDRR Format    │  │
│  │ Capabilities│  │  │  │ Loader      │  │  │  │  (64MB shards)  │  │
│  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────────┘  │
│  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────────┐  │
│  │ Buffer Pool │  │  │  │ Dequantize  │  │  │  │ Shard Manager   │  │
│  └─────────────┘  │  │  │ Q4_K→F32    │  │  │  │ (OPFS/Bridge)   │  │
│  ┌─────────────┐  │  │  └─────────────┘  │  │  └─────────────────┘  │
│  │ WGSL Kernels│  │  │                   │  │  ┌─────────────────┐  │
│  │ (27 files)  │  │  │                   │  │  │   Downloader    │  │
│  └─────────────┘  │  │                   │  │  └─────────────────┘  │
└───────────────────┘  └───────────────────┘  └───────────────────────┘
```

## Design Philosophy

DOPPLER makes deliberate architectural tradeoffs that diverge from pre-compiled approaches like WebLLM/TVM. These enable capabilities that are impossible with monolithic compiled models.

### Key Principles

| Principle | Implementation | Why |
|-----------|----------------|-----|
| **Code/Data Separation** | Generic WGSL kernels + weight shards | Enables P2P distribution, expert paging, LoRA hot-swap |
| **GPU Fusion** | All tensor ops stay on GPU | Makes JS vs WASM irrelevant (0.5ms vs 25ms GPU time) |
| **Minimal Readback** | Only final logits read to CPU | Avoids 2-6ms GPU→CPU transfer per readback |
| **JavaScript Orchestration** | JS dispatches GPU work, handles sampling | Debugging, rapid iteration, WebRTC P2P integration |

### GPU Fusion in Practice

```
Per-token decode step:
├─ GPU compute:     25ms   ████████████████████████████ (96%)
├─ JS orchestration: 0.5ms ██ (2%)
└─ Logits readback:  0.5ms ██ (2%)

WASM would make 0.5ms faster → irrelevant when GPU is 96% of time
```

### Readback Minimization

| Operation | Location | Readback? |
|-----------|----------|-----------|
| Q/K/V projections | GPU matmul | No |
| Attention scores | GPU fused kernel | No |
| MoE router decisions | GPU softmax+topk | No |
| Expert FFN outputs | GPU matmul | No |
| Intermediate hidden states | GPU buffers | No |
| **Final logits** | **GPU → CPU** | **Yes (unavoidable)** |

Only ONE readback per generated token. This is the key performance insight.

### Capability vs Performance Tradeoff

DOPPLER accepts ~20% kernel performance gap vs TVM auto-tuned kernels because it enables:

| Capability | Why Impossible with TVM |
|------------|------------------------|
| 90GB MoE on 8GB VRAM | Expert paging requires dynamic buffer binding |
| P2P shard distribution | Can't split compiled binary across peers |
| LoRA hot-swap | Compiled model can't change weights at runtime |
| Speculative decoding | Coordinating two compiled models is awkward |

See [VISION.md](VISION.md#architectural-bets) for detailed rationale and concrete examples.

---

## Module Structure

| Directory | Purpose |
|-----------|---------|
| `gpu/` | WebGPU device, buffer pool, WGSL kernels |
| `inference/` | Pipeline orchestration, KV cache, MoE routing |
| `loader/` | Weight loading, dequantization |
| `storage/` | RDRR format, OPFS shard management |
| `memory/` | Memory64, unified memory detection |
| `tools/` | Conversion CLI, quantizer |
| `bridge/` | Native Bridge for local file access |

---

## 1. GPU Subsystem (`gpu/`)

### device.ts - WebGPU Initialization

Initializes WebGPU with capability detection:

```javascript
// Feature flags detected at init
{
  hasF16: boolean,        // shader-f16 extension
  hasSubgroups: boolean,  // subgroups extension
  hasTimestampQuery: boolean,
  maxBufferSize: number,
  maxStorageBufferBindingSize: number,
}
```

**Adapter Selection Strategy:**
1. High-performance adapter (discrete GPU)
2. Low-power adapter (integrated GPU)
3. Any available adapter

### buffer-pool.ts - GPU Buffer Pooling

Power-of-2 bucket pooling to avoid allocation churn:

```
Bucket sizes: 256B, 512B, 1KB, 2KB, ... 256MB
acquireBuffer(size) → finds smallest bucket >= size
releaseBuffer(buf) → returns to pool for reuse
```

Key insight: WebGPU buffer allocation is expensive (~1ms), pooling amortizes this.

### kernel-selector.ts - Kernel Dispatch

Routes operations to optimal kernel based on capabilities:

```javascript
// Example: matmul routing
if (hasF16 && weightsAreF16) → matmul_f16.wgsl
else if (hasF16 && weightsAreF16 && activationsAreF32) → matmul_f16w_f32a.wgsl
else → matmul_f32.wgsl
```

Auto-tuning: Benchmarks kernel variants at startup, caches best choice per device.

### WGSL Kernels (`gpu/kernels/`)

| Kernel | Description | Key Features |
|--------|-------------|--------------|
| **attention.wgsl** | Fused MHA | Flash Attention, online softmax, GQA |
| **attention_streaming.wgsl** | Large context | Streaming for >8K sequences |
| **attention_small.wgsl** | Short context | Optimized for decode (queryLen=1) |
| **matmul_f32.wgsl** | FP32 tiled matmul | 16x16 tiles, shared memory |
| **matmul_f16.wgsl** | FP16 tiled matmul | F32 accumulator for stability |
| **matmul_f16w_f32a.wgsl** | Mixed precision | F16 weights, F32 activations |
| **dequant_shared.wgsl** | Q4_K→F32 | llama.cpp format, workgroup |
| **dequant_subgroup.wgsl** | Q4_K→F32 | Subgroup shuffle optimization |
| **dequant_f16_out.wgsl** | Q4_K→F16 | Direct F16 output |
| **dequant_mxfp4.wgsl** | MXFP4→F32 | GPT-OSS MoE experts |
| **rmsnorm.wgsl** | RMS normalization | Per-token normalization |
| **softmax.wgsl** | Online softmax | Numerically stable |
| **rope.wgsl** | Rotary embeddings | Precomputed frequencies |
| **silu.wgsl** | SiLU activation | x * sigmoid(x) |
| **swiglu.wgsl** | SwiGLU | Fused gate*up + down |
| **topk.wgsl** | Top-k selection | For sampling |
| **gather.wgsl** | Embedding lookup | Token→hidden |
| **moe_gather.wgsl** | MoE token gather | Batch tokens to experts |
| **scatter_add.wgsl** | MoE combine | Combine expert outputs |
| **bf16_to_f32.wgsl** | BF16 conversion | For SafeTensors |
| **cast_f32_to_f16.wgsl** | Downcast | VRAM reduction |
| **bias_add.wgsl** | Add bias | For linear layers |
| **residual.wgsl** | Residual add | Skip connections |

---

## 2. Inference Pipeline (`inference/`)

### pipeline.ts - Main Orchestration

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

**Key Pipeline Features:**
- GPU-native: No CPU readback until final logits sampling
- GQA support: Multiple Q heads share K,V heads
- Sandwich norms: Gemma 3 pre/post FFN norms
- YARN RoPE: Extended context via per-dimension scaling

### Pipeline Submodules (`inference/pipeline/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.ts` | 325 | Pipeline configuration, model params |
| `sampling.ts` | 203 | Token sampling (top-k, top-p, temperature) |
| `generate.ts` | 279 | Generation loop helpers |
| `layer.ts` | 180 | Per-layer processing |
| `prefill.ts` | 131 | Prompt prefill phase |
| `decode.ts` | 144 | Autoregressive decode phase |
| `embed.ts` | 173 | Token embedding |
| `stats.ts` | 174 | Performance statistics |
| `stopping.ts` | 178 | Stop condition detection |
| `index.ts` | 150 | Module exports |

**Note:** These modules are split for maintainability but the main `pipeline.ts` still uses internal methods. Full wiring is in progress.

### kv-cache.ts - KV Cache Management

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
- `paged`: Block-based (future, for very long contexts)

**KV dtype:** F16 when supported, halves VRAM usage.

### tokenizer.ts - Tokenization

Loads tokenizer from:
1. Bundled `tokenizer.json` in model directory
2. HuggingFace-format vocab files

Supports chat templates via `tokenizer_config.json`.

### moe-router.ts - Mixture of Experts

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

## 3. Loader (`loader/doppler-loader.ts`)

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

**Gemma 3 norm offset:** RMSNorm uses `(1 + weight) * x` instead of `weight * x`. DopplerLoader applies +1 offset during load for SafeTensors source (GGUF has it baked in).

---

## 4. Storage (`storage/`)

### RDRR Format (`rdrr-format.ts`)

Custom model format optimized for browser streaming:

```
model-directory/
├── manifest.json          # Tensor locations, config, hashes
├── shard_000.rdrr         # 64MB shard
├── shard_001.rdrr
├── ...
└── tokenizer.json         # Optional bundled tokenizer
```

**manifest.json structure:**
```json
{
  "modelId": "gemma-3-1b-q4",
  "version": "1.0",
  "quantization": "Q4_K_M",
  "totalSize": 1073741824,
  "hashAlgorithm": "blake3",
  "config": { /* HuggingFace config */ },
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
    { "filename": "shard_000.rdrr", "size": 67108864, "hash": "..." }
  ]
}
```

### shard-manager.ts - OPFS Integration

Uses Origin Private File System for persistent model storage:
- `initOPFS()` - Initialize root directory
- `loadShard(idx)` - Read shard to ArrayBuffer
- `verifyIntegrity()` - Check all shard hashes
- `computeHash(data, algo)` - Blake3/SHA256

### downloader.ts - Model Download

Streaming download with:
- Progress callbacks
- Shard-by-shard integrity verification
- Resume support (partial downloads)

---

## 5. Memory Subsystem (`memory/`)

### capability.ts - Memory Detection

Detects runtime capabilities:
```javascript
{
  hasMemory64: boolean,     // WebAssembly Memory64
  maxHeapSize: number,      // JS heap limit
  isUnifiedMemory: boolean, // Apple Silicon
}
```

### unified-detect.ts - Unified Memory Detection

Apple Silicon detection for optimal buffer sharing:
- Unified memory allows larger models (no PCIe copy)
- Detected via `navigator.gpu.requestAdapter()` heuristics

---

## 6. Tools (`tools/`)

### convert-cli.ts - Model Conversion

Converts HuggingFace models to RDRR format:
```bash
npx tsx convert-cli.ts \
  --input ./hf-model \
  --output ./rdrr-model \
  --quantize Q4_K_M
```

### quantizer.ts - Q4_K Quantization

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

**Post-mortem note:** Early bug stored `min` with different sign convention, causing all dequantized values to be positive. See `docs/GEMMA3-DEBUG-POSTMORTEM.md`.

---

## 7. Provider API (`doppler-provider.ts`)

Public API for LLM client integration:

```javascript
// Initialize
await initDoppler();

// Load model
await loadModel('gemma-3-1b-q4', modelUrl, onProgress);

// Generate (streaming)
for await (const token of generate(prompt, options)) {
  console.log(token);
}

// Chat interface
const response = await dopplerChat(messages, options);
```

**Capability Tiers:**
| Tier | Memory | Max Model |
|------|--------|-----------|
| 1 | Unified (Apple Silicon) | 60GB |
| 2 | Memory64 | 40GB MoE |
| 3 | Basic | 8GB small MoE |

---

## Data Flow: Single Token Generation

For detailed kernel-level execution trace including tensor shapes, kernel selection, and fusion analysis, see [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md).

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
All tensor operations stay on GPU until final sampling. This minimizes CPU↔GPU transfers which are the primary bottleneck in browser WebGPU.

### 2. Q4_K Quantization
4-bit quantization reduces model size 4x while maintaining quality. The llama.cpp Q4_K format is battle-tested and well-documented.

### 3. 64MB Shards
Shard size balances:
- Small enough for reliable streaming download
- Large enough to minimize request overhead
- Aligned with OPFS block allocation

### 4. Streaming Weight Load
Large tensors (embeddings, LM head) are streamed directly to GPU buffers to avoid JS heap exhaustion.

### 5. Capability-Based Kernel Selection
Different devices get different kernel implementations:
- F16 hardware → F16 kernels for 2x throughput
- Subgroup support → shuffle-based reductions
- Large context → streaming attention

See [EXECUTION_PIPELINE.md Part III](EXECUTION_PIPELINE.md#part-iii-capability-based-kernel-selection) for complete kernel selection decision trees and RDRR runtime hints.

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `doppler-provider.ts` | 726 | Public API, LLM client integration |
| `inference/pipeline.ts` | 3884 | Main inference orchestration |
| `inference/kv-cache.ts` | 953 | KV cache management |
| `inference/tokenizer.ts` | 1485 | Tokenization wrapper |
| `inference/moe-router.ts` | 627 | MoE expert routing |
| `loader/doppler-loader.ts` | 1204 | Weight loading, dequant |
| `gpu/device.ts` | 330 | WebGPU initialization |
| `gpu/kernel-selector.ts` | 27 | Kernel dispatch (routing) |
| `gpu/kernel-tuner.ts` | ~700 | Auto-tuning benchmarks |
| `gpu/buffer-pool.ts` | 506 | Buffer pooling |
| `storage/rdrr-format.ts` | 363 | RDRR format parsing |
| `storage/shard-manager.ts` | 764 | OPFS shard management |
| `tools/quantizer.ts` | 349 | Q4_K quantization |
| `tools/convert-cli.ts` | 599 | Model conversion CLI |

---

## Related Documentation

- `docs/GEMMA3-DEBUG-POSTMORTEM.md` - Q4_K quantizer bug analysis
- `roadmap/PHASE_4_P2P.md` - P2P model distribution
- `docs/spec/RDRR_FORMAT.md` - RDRR format specification

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

