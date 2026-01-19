# GPU Subsystem Internals
## 1. GPU Subsystem (`gpu/`)

### device.js - WebGPU Initialization

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

### memory/buffer-pool.js - GPU Buffer Pooling

Power-of-2 bucket pooling to avoid allocation churn:

```
Bucket sizes: 256B, 512B, 1KB, 2KB, ... 256MB
acquireBuffer(size) → finds smallest bucket >= size
releaseBuffer(buf) → returns to pool for reuse
```

Key insight: WebGPU buffer allocation is expensive (~1ms), pooling amortizes this.

### buffer-dtypes.js - Buffer Metadata

Tracks per-buffer dtype and layout metadata so kernels can select correct execution paths.

### kernel-selection-cache.js - Kernel Selection Cache

Caches kernel selections and warm status to avoid repeated benchmarking on the same device.

### gpu/kernels/*.js - Kernel Dispatch

Routes operations to optimal kernel based on capabilities:

```javascript
// Example: matmul routing (matmul.js)
if (hasF16 && weightsAreF16) → matmul_f16.wgsl
else if (hasF16 && weightsAreF16 && activationsAreF32) → matmul_f16w_f32a.wgsl
else → matmul_f32.wgsl
```

Auto-tuning: Benchmarks kernel variants at startup, caches best choice per device.

### profiler.js - GPU Profiling

Optional marker-based profiling to collect per-op timings during debug and tuning.

### partitioned-buffer-pool.js - Multi-Model Buffer Pools

Partitions buffer pools by model/expert to reduce contention during multi-model execution.

### multi-model-recorder.js - Shared Prefix Recording

Records command streams across multiple models to reuse shared prefix KV and reduce overhead.

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


## Part III: Capability-Based Kernel Selection

DOPPLER dynamically selects kernel variants at runtime based on GPU capabilities and model configuration. This section documents the complete selection pipeline.

### GPU Capability Detection

At initialization (`gpu/device.js`), DOPPLER probes the WebGPU adapter for available features:

```typescript
interface KernelCapabilities {
  hasF16: boolean;              // shader-f16 extension
  hasSubgroups: boolean;        // subgroups extension (shuffle ops)
  hasSubgroupsF16: boolean;     // subgroups-f16 (combined)
  hasTimestampQuery: boolean;   // GPU profiling
  maxBufferSize: number;        // Max storage buffer (bytes)
  maxWorkgroupSize: number;     // Max threads per workgroup
  maxWorkgroupStorageSize: number;  // Shared memory limit (bytes)
  adapterInfo: {
    vendor: string;             // "apple", "nvidia", "amd", etc.
    architecture: string;       // "common-3", "ampere", etc.
    device: string;             // GPU model name
  };
}
```

**Feature Detection Flow:**

```
navigator.gpu.requestAdapter()
       │
       ▼
Probe adapter.features for:
  - 'shader-f16'      → enables F16 matmul, F16 KV cache
  - 'subgroups'       → enables subgroup shuffle reductions
  - 'timestamp-query' → enables GPU-side profiling
       │
       ▼
Request device with detected features
       │
       ▼
Cache capabilities in kernelCapabilities global
```

### Kernel Configuration Schema

All kernel variants are defined in `gpu/kernels/utils.js` as `KERNEL_CONFIGS`:

```typescript
interface KernelConfig {
  shaderFile: string;                        // WGSL file name
  entryPoint: string;                        // Function to call
  workgroupSize: [number, number, number];   // Default workgroup dims
  requires: string[];                        // Required GPU features
  validate?: (seqLen, numHeads, headDim) => void;  // Optional limits check
}
```

**Example - Matmul Variants:**

| Variant | Shader File | Requirements | Use Case |
|---------|-------------|--------------|----------|
| `f32` | `matmul_f32.wgsl` | none | Fallback for all GPUs |
| `f16` | `matmul_f16.wgsl` | `shader-f16` | Both inputs F16 |
| `f16w_f32a` | `matmul_f16w_f32a.wgsl` | `shader-f16` | F16 weights, F32 activations |
| `gemv` | `matmul_gemv.wgsl` | `shader-f16` | M=1 decode, basic |
| `gemv_subgroup` | `matmul_gemv_subgroup.wgsl` | `shader-f16`, `subgroups` | M=1 decode, optimized |
| `q4_fused` | `matmul_q4_fused.wgsl` | `shader-f16`, `subgroups` | Fused Q4_K dequant+matmul |

### Selection Decision Trees

#### Matmul Kernel Selection

```
selectMatmulKernel(aDtype, bDtype, M, outputDtype)
       │
       ├── bDtype == 'q4k'?
       │       │
       │       ├── M == 1 → 'q4_fused' (GEMV, fused dequant)
       │       └── M > 1  → 'q4_fused_batched' (tiled, fused dequant)
       │
       ├── M == 1 && bDtype == 'f16' && aDtype == 'f32'?
       │       │
       │       ├── hasSubgroups → 'gemv_subgroup' (1.5x faster)
       │       └── else         → 'gemv'
       │
       ├── aDtype == 'f16' && bDtype == 'f16' && hasF16?
       │       │
       │       └── outputDtype == 'f16' → 'f16'
       │
       ├── bDtype == 'f16' && aDtype == 'f32' && hasF16?
       │       │
       │       └── → 'f16w_f32a' (mixed precision)
       │
       └── else → 'f32' (universal fallback)
```

#### Attention Kernel Selection

```
selectAttentionKernel(headDim, kvDtype, phase)
       │
       ├── headDim <= 64 && sharedMem >= 49KB?
       │       │
       │       └── Tiled attention (fastest, fits in shared memory)
       │           ├── kvDtype == 'f16' → 'prefill_f16kv' / 'decode_f16kv'
       │           └── else             → 'prefill' / 'decode'
       │
       ├── headDim <= 256 && sharedMem >= 4KB?
       │       │
       │       └── Small tiled attention
       │           ├── kvDtype == 'f16' → 'prefill_small_f16kv' / 'decode_small_f16kv'
       │           └── else             → 'prefill_small' / 'decode_small'
       │
       └── else (headDim > 256)
               │
               └── Streaming attention (processes KV in blocks)
                   ├── kvDtype == 'f16' → 'prefill_streaming_f16kv' / 'decode_streaming_f16kv'
                   └── else             → 'prefill_streaming' / 'decode_streaming'

Note: Gemma 3 1B has headDim=256, so it uses the small tiled attention path.
```

#### Dequantization Kernel Selection

```
selectDequantKernel(outputDtype)
       │
       ├── hasSubgroups && outputDtype == 'f16' && hasF16?
       │       │
       │       └── 'subgroup_f16out' (fastest)
       │
       ├── hasSubgroups?
       │       │
       │       └── 'subgroup' (uses shuffle for reduction)
       │
       ├── outputDtype == 'f16' && hasF16?
       │       │
       │       └── 'shared_f16out'
       │
       └── else → 'shared' (universal fallback)
```

### Model Config → Kernel Mapping

Model architecture parameters directly influence kernel selection:

| Model Parameter | Kernel Impact |
|-----------------|---------------|
| `headDim` | Determines attention tier (tiled vs streaming) |
| `numKVHeads` | Affects KV cache size, F16 cache viability |
| `intermediateSize` | FFN matmul dimensions |
| `vocabSize` | LM head matmul size (often largest operation) |
| `quantization` | Selects dequant kernel, fused Q4K matmul |

**Example - Gemma 3 1B:**

```
headDim = 256  →  Uses small tiled attention (fits in 4KB shared memory)
                  Selects 'attention_small.wgsl' / 'attention_small_f16kv.wgsl'

quantization = 'Q4_K_M'  →  Uses 'q4_fused' for decode GEMV
                            Uses 'q4_fused_batched' for prefill

numKVHeads = 1 (GQA)  →  Small KV cache, enables F16 KV
                        Only 256 floats per token per layer
```

### Kernel Path Overrides

The manifest can include kernel path overrides to select explicit kernel dispatch sequences:

```json
{
  "optimizations": {
    "kernelPath": "gemma2-q4k-fused-f16a"
  },
  "inference": {
    "defaultKernelPath": "gemma2-q4k-dequant-f16a"
  }
}
```

Runtime config can override with `runtime.inference.kernelPath`. See `CONFIG.md` for available paths and structure.

### Auto-Tuning System

DOPPLER includes an auto-tuning system (`gpu/kernel-tuner.js`) that benchmarks kernel variants at runtime:

```typescript
// Tune kernels for specific model config (Gemma 3 1B example)
const results = await autoTuneKernels({
  hiddenSize: 1152,
  intermediateSize: 6912,
  numHeads: 4,
  headDim: 256,
  vocabSize: 262144,
});

// Results cached in IndexedDB for future sessions
// Format: { operation: { optimalWorkgroupSize, variantTimings } }
```

**Tuning Flow:**

```
1. For each kernel operation (matmul, attention, rmsnorm, etc.):
   │
   ├── Generate test inputs matching model config
   │
   ├── Run each compatible variant N times (default: 10)
   │
   ├── Measure median execution time
   │
   └── Cache optimal variant and workgroup size

2. On subsequent runs:
   │
   └── Load cached results, skip benchmarking
```

**Manual Tuning:**

```typescript
import { getTunedWorkgroupSize } from './gpu/kernels/index.js';

// Get optimal workgroup size for matmul with specific dimensions (Gemma 3 example)
const [wgX, wgY, wgZ] = await getTunedWorkgroupSize('matmul', {
  M: 1,
  N: 4096,
  K: 1152,
});
```

### Kernel Prewarm

To avoid shader compilation stalls during inference, DOPPLER can prewarm all compatible kernels at startup:

```typescript
import { prewarmKernels } from './gpu/kernels/index.js';

// Compile all kernels that the current GPU supports
await prewarmKernels();

// Output: "[KernelSelector] Prewarmed 47 kernel pipelines"
```

This is especially important for:
- First inference after page load
- Mobile GPUs with slow shader compilation
- WebGPU implementations with synchronous compile

### Capability Tiers

DOPPLER defines capability tiers for common GPU classes:

| Tier | Example GPUs | Features | Typical Kernels |
|------|--------------|----------|-----------------|
| **Tier 1** | Apple M1+, RTX 30+ | F16, subgroups | `gemv_subgroup`, `q4_fused`, streaming F16KV |
| **Tier 2** | Intel Xe, AMD RDNA2+ | F16, no subgroups | `gemv`, `f16w_f32a`, streaming F16KV |
| **Tier 3** | Older Intel, mobile | No F16 | `f32`, shared dequant, F32 KV |

**Detection:**

```typescript
function getCapabilityTier(caps: KernelCapabilities): 1 | 2 | 3 {
  if (caps.hasF16 && caps.hasSubgroups) return 1;
  if (caps.hasF16) return 2;
  return 3;
}
```

### Debugging Kernel Selection

Enable kernel trace output to see kernel selection decisions:

```bash
# CLI (preferred): trace only kernels
npm run debug -- --config debug

# Config-driven (repeatable)
npm run debug -- --config '{"runtime":{"inference":{"debug":{"trace":{"enabled":true,"categories":["kernels"]}}}}}'
```

Use config-only kernel path overrides for testing (see `style/WGSL_STYLE_GUIDE.md`).

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
