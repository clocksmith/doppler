# Phase 1: Performance Parity

**Status:** In Progress
**Prerequisites:** None (foundational)
**Goal:** Match or beat WebLLM performance for dense models.

---

## Milestones

### Completed ✅
- [x] Gemma 3 1B working E2E ✅ Dec 2025
- [x] Tiled matmul optimization ✅
- [x] FlashAttention-style fusion ✅
- [x] W4A16 quantized matmul ✅
- [x] Fused decode path (layers+logits+sampling) ✅ Dec 2025
- [x] GPU sampling (argmax + top-k on GPU) ✅ Dec 2025
- [x] GPU top-k sampling integration ✅ Dec 20 2025
- [x] Gate+Up FFN fusion ✅ Dec 2025
- [x] Command buffer batching ✅ Dec 2025
- [x] Column-wise Q4K layout ✅ Dec 2025

### In Progress
- [ ] 40+ tok/s decode on Gemma 3 1B (P0) - currently **~4 tok/s (10x gap)** ⚠️
  - [x] Attention decode kernel (subgroup) - ✅ Implemented Dec 20 (expect 2-3x overall)
  - [x] GPU-only decode loop infrastructure - ✅ Implemented Dec 20 (needs integration)
  - [x] Test & benchmark new kernels (test:kernels:perf)
  - [x] Fix fused Q4K thread utilization - P0 (recover 2.7x)
- [x] GPU timestamp profiling to identify bottleneck (P0) - metrics now populate `gpu_time_ms_*`
- [ ] Investigate 50% performance regression (P0) - was 8 tok/s, now 4 tok/s (logits/attention logging + embed/logits batching fixed, re-benchmark)
- [ ] Llama 3.2 models validated (P0) - 1B done, 3B pending

### Kernel Fusion Roadmap (See 1.14)
- [ ] Full F16 activation pipeline (P0) - 1.5-2x speedup
- [x] Residual+RMSNorm fusion (P0) - ✅ Done Dec 2025 - **~1.5% actual impact** (RMSNorm <1% GPU time)
- [ ] Quantized Matmul+RMSNorm fusion (P0) - 1.2-1.5x speedup
- [ ] Matmul+RMSNorm epilogue (P1) - 1.1-1.3x speedup
- [ ] Attention+Residual fusion (P1) - 1.1-1.2x speedup
- [ ] Matmul+SiLU epilogue (P2) - 1.1-1.2x speedup
- [ ] Parallel FFN kernels (P2) - 1.1-1.3x speedup
- [ ] Parallel Q/K/V projection (P2) - 1.1-1.2x speedup

### Speedup Optimizations Roadmap (See 1.15, 1.20, 1.22)
- [x] Fix fused Q4K thread utilization (P0) - recover 2.7x
- [x] Complete workgroup auto-tuning (P1) - 1.1-1.2x speedup
- [x] **GPU-only decode loop (P1) - 2-5x speedup** (See 1.20) — ✅ Integrated in `inference/pipeline.ts` batch path
- [x] **Attention decode kernel (P0) - 3.6-7.2x attention speedup** (See 1.22) — ✅ Implemented Dec 20, ready for testing
- [ ] Shader constants → uniforms migration (P1) - config fidelity
- [ ] Remove F32 intermediate allocations (P1) - 1.1-1.2x speedup
- [ ] Speculative decoding (P2) - 2-3x speedup
- [ ] Tensor parallelism (P2) - 2x per GPU

---

## Work Items

### 1.1 WeInfer Optimizations (Critical Path)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Command buffer batching | P0 | ✅ Done | `gpu/command-recorder.ts` |
| Buffer reuse strategy | P0 | ✅ Done | `gpu/buffer-pool.ts` |
| GPU-side sampling | P0 | ✅ Done | Fused argmax, reads 4 bytes/token (was 1MB) |
| Fused decode path | P0 | ✅ Done | Single submit for layers+logits+argmax |
| Deferred result fetching | P0 | ✅ Done | `inference/pipeline/logits.ts` |
| Async pipeline | P0 | ✅ Done | Weights pre-loaded |
| Gate debug-only generation logging | P0 | ✅ Done | Avoids token decode + top-k work in hot path |
| Batch decode path cleanup | P0 | ✅ Done | Remove unused buffers + redundant dynamic imports |

### 1.2 Kernel Infrastructure

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Shader prewarm during load | P0 | ✅ Done | |
| F16 weight storage | P0 | ✅ Done | Mixed-precision matmul |
| KV cache f16 allocation | P0 | ✅ Done | F16 attention path |
| Multi-tier attention kernels | P0 | ✅ Done | Large/small/streaming tiers |
| SwiGLU fused activation | P1 | ✅ Done | Gate + up + SiLU in one pass |
| Kernel auto-tuner | P1 | ✅ Done | `kernel-tuner.ts` with localStorage |
| Speculative decoding framework | P2 | ✅ Done | Needs draft model wiring |
| Kernel benchmark harness | P0 | ✅ Done | Playwright eval shim for __name in CLI |

### 1.3 Kernel Optimizations (Performance)

| Task | Priority | Status | Impact | Notes |
|------|----------|--------|--------|-------|
| Tiled matmul optimization | P0 | ✅ Done | 2-3x | 16x16 shared memory tiles in `matmul_f32.wgsl` |
| Subgroup operations | P0 | ✅ Done | 1.5x | `matmul_gemv_subgroup.wgsl` |
| Multi-column GEMV (LM head) | P0 | ✅ Done | ~0% | 32 cols/wg in `matmul_gemv_subgroup.wgsl` - LM head not bottleneck |
| Workgroup size auto-tuning | P1 | ✅ Done | 1.2-1.5x | Benchmarks added for attention/softmax/rmsnorm/dequant in `kernel-tuner.ts` |
| FlashAttention-style fusion | P1 | ✅ Done | 2x | Tiled + online softmax, 3 device-aware tiers |
| Fused FFN kernel (gate+up weights) | P0 | ✅ Done | 1.2-1.3x | 3→2 passes via gate+up weight concatenation |
| Kernel uniform audit (vs constants) | P0 | ⬜ TODO | Config fidelity | Catalog kernels with baked constants (e.g. `gpu/kernels/attention*.wgsl`, `matmul_q4_fused.wgsl`), add manifest-sourced uniforms so `tools/convert-cli.ts` → `storage/rdrr-format.ts` → runtime config stay in sync |
| Matmul+SiLU epilogue fusion | P2 | ⬜ TODO | 1.1-1.2x | 2→~1.5 passes, fuse first matmul with split+SiLU |
| Full f16 activation pipeline | P0 | ⏳ Partial | 1.5-2x | F16 KV cache done; F32 activations intentional |
| W4A16 quantized matmul | P0 | ✅ Done | 2-3x | Fused Q4K kernel in `matmul_q4_fused.wgsl` |

### 1.4 Model Validation

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Gemma 3 1B E2E | P0 | ✅ Done | Dec 2025 |
| Gemma 3 4B E2E | P0 | ⬜ TODO | Same arch as 1B |
| Gemma 3 variant regression tests (8 SKUs) | P0 | ⬜ TODO | Re-run suites for the variants whose weights changed |
| Llama 3.2 1B E2E | P0 | ✅ Done | Bench run (xs, 2025-12-28) |
| Llama 3.2 3B E2E | P0 | ⬜ TODO | Standard Llama |
| Llama 3.1 8B E2E | P1 | ⬜ TODO | Needs unified mem |
| Mistral 7B E2E | P1 | ⬜ TODO | Standard Llama-like |
| Qwen 2 E2E | P1 | ⬜ TODO | Different tokenizer |
| Validate 8GB model load | P0 | ⬜ TODO | Memory test |
| Validate 16GB model load | P1 | ⬜ TODO | Large model test |

### 1.5 Performance Roadmap (8 → 40+ tok/s)

**Current:** ~8 tok/s decode on Gemma 3 1B (M3) - Dec 2025 benchmark (column_wise Q4K)
**Target:** ≥40 tok/s decode (5x improvement needed)

| Optimization | Est. Speedup | Cumulative | Priority | Status |
|--------------|--------------|------------|----------|--------|
| Column-wise Q4K layout | **2.7x** | 2.7x | P0 | ✅ Done (`--q4k-layout column_wise` default) |
| Fused Q4K matmul | ~~1.3-1.5x~~ | ~~2-3x~~ | ~~P0~~ | ❌ **SLOWER** (see 1.7) |
| F16 KV cache auto-detect | - | - | P0 | ✅ Done (init.ts auto-selects F16 when supported) |
| BF16→F16 matmul weights | 1.2-1.5x | 2.5-4x | P0 | ✅ Done (spans path fixed Dec 2025) |
| FFN gate+up fusion (3→2 passes) | 1.2-1.3x | 3-5x | P0 | ✅ Done |
| GPU sampling (no logit readback) | 1.3-1.5x | 4-6x | P0 | ✅ Done (fused decode path) |
| Multi-column LM head GEMV | ~0% | - | P0 | ✅ Done (not bottleneck - see 1.11) |
| GPU timestamp profiling | - | - | P0 | ✅ Done (metrics in `gpu_time_ms_*`) |
| Complete workgroup auto-tuning | 1.1-1.2x | 4.5-7x | P1 | ✅ Done (kernel-tuner loops for attention/softmax/rmsnorm/dequant) |
| Speculative decoding | 2-3x | 9-21x | P2 | ⬜ Framework ready |
| Streaming prefill | - | - | P2 | ⬜ TODO (for long contexts) |

**Note:** Need to re-convert models with `--transpose-weights` flag to benefit from column-major layout.

**Performance progression (Dec 2025):**
| Stage | tok/s | Key Change |
|-------|-------|------------|
| Baseline (debug mode) | 2 | CPU sampling for first 5 tokens |
| + GPU sampling fix | 3.3 | Removed `!isDebugStep` gate |
| + Command batching | 4 | Set `debug: false` in benchmark |
| + Fused decode path | 7 | Single submit for layers+logits+argmax |

**Latest benchmark (2025-12-28, xs prompt, llama-3.2-1b-instruct-q4):**
- TTFT 438 ms
- Prefill 18 tok/s
- Decode 19 tok/s

**Remaining gap analysis:**
- WebLLM achieves ~24 ms/token vs DOPPLER ~140 ms/token (6x gap)
- LM head multicol optimization had no effect → bottleneck is elsewhere
- GPU timestamps now available to identify which kernels dominate per forward pass

### 1.6 Column-Major Weight Storage

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Add `--transpose-weights` flag to rdrr-writer | P0 | ✅ Done | `transposeWeights` option in writer |
| Store layout metadata in manifest | P0 | ✅ Done | `weightsTransposed: true` in manifest |
| Update matmul kernel selection | P0 | ✅ Done | `matmul.ts` handles `transposeB` based on layout |
| Re-convert test models with column layout | P0 | ✅ Done | Use `--transpose-weights` flag |
| **Row-wise Q4K quantization** | P0 | ✅ **Done** | `--q4k-layout row_wise` (default) in convert-cli |
| Column-wise Q4K (batched matmul only) | P2 | ⬜ TODO | Only helps prefill, not decode |

**Implementation:** `rdrr-writer.ts` has `transposeWeights` option that pre-transposes matmul weights. Manifest stores `weightsTransposed: true`. Loader sets `transposeB: false` when weights are pre-transposed.

#### Why Column-Major is Faster (GPU Coalescing)

**The operation:** `output[1, out] = input[1, K] @ weight[out, K]^T`

When threads in a GPU warp access consecutive memory addresses, the hardware coalesces into a single transaction. Strided access splits into multiple transactions → high latency.

```
Row-major W[out, K]:
Thread 0 reads W[0, 0]    ← address 0
Thread 1 reads W[1, 0]    ← address K (strided - BAD)
Thread 2 reads W[2, 0]    ← address 2K

Column-major W^T[K, out]:
Thread 0 reads W^T[0, 0]  ← address 0
Thread 1 reads W^T[0, 1]  ← address 1 (contiguous - GOOD)
Thread 2 reads W^T[0, 2]  ← address 2
```

| Layout | Memory Pattern | GPU Coalescing | Performance |
|--------|----------------|----------------|-------------|
| Row-major W[out, K] | Row i contiguous | Threads read strided | Slower |
| Column-major W^T[K, out] | Column i contiguous | Threads read contiguous | **1.5-2x faster** |

#### Current State by Format

| Format | Current Layout | Optimal Layout | Status |
|--------|---------------|----------------|--------|
| F16/BF16 | Column-major ✅ | Column-major | Done |
| Q4K | **Column-wise ✅** | Column-wise | **BENCHMARKED** (converter uses `--q4k-layout column_wise` by default) |

#### Q4K Block Layout Problem

Q4K has 256-value super-blocks with embedded metadata:
```
Block (144 bytes): [d: f16, dmin: f16, scales: 12B, nibbles: 128B]
```

**Current (flat packed):** Blocks cross row boundaries
```
Flat: [blk0][blk1][blk2][blk3][blk4][blk5]...
       ←─row 0──→←─row 0──→←row 1→←─row 1──→  ← WRONG!
```

**Column-wise Q4K:** Blocks organized by input column - **BENCHMARKED FASTEST ✅**
```
Col 0: [blk0][blk5][blk10]...  ← All K positions for output 0
Col 1: [blk1][blk6][blk11]...  ← All K positions for output 1
```

#### Q4K Layout Benchmark Results (Dec 2025)

| Layout | Decode tok/s | vs Baseline | Notes |
|--------|--------------|-------------|-------|
| **column_wise** | **8.0** | +14% | **DEFAULT - FASTEST** |
| flat | 7.0 | baseline | Simple packing |
| row_wise | 3.0 | -57% | Fused kernel has poor thread utilization |

**Status:** ✅ Benchmarked in `tools/quantizer.ts` and `tools/convert-cli.ts`

**Available functions:**
- `quantizeToQ4KMColumnWise(data, shape)` - Column-aligned Q4K blocks **(DEFAULT)**
- `quantizeToQ4KMRowWise(data, shape)` - Row-aligned Q4K blocks
- `quantizeToQ4KM(data, shape)` - Flat sequential packing
- `getQ4KSize(shape, layout)` - Calculate expected Q4K size

**Usage:**
```bash
# Convert with column-wise Q4K (default - fastest for GEMV decode)
npx tsx doppler/tools/convert-cli.ts model/ output/ --quantize q4_k_m

# Explicitly specify layout
npx tsx doppler/tools/convert-cli.ts model/ output/ --quantize q4_k_m --q4k-layout column_wise
```

**Why column-wise is fastest:**

For GEMV (decode with M=1), computing `C[1, N] = A[1, K] × B[K, N]`:
- Each output column needs to read ALL K weights for that column
- Column-wise packing: column j's blocks are **contiguous in memory**
- Dequant kernel reads contiguous blocks → coalesced GPU access → high bandwidth

**Row-wise is SLOWER because:**
- The fused Q4K kernel has 256 threads per workgroup
- For K=1152, there are only 5 Q4K blocks per row
- **251 of 256 threads are IDLE** → massive underutilization
- See section 1.7 for details

**References:**
- [NVIDIA Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [WebGPU Matmul 1TFLOP Optimization](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
- [llama.cpp K-Quants Discussion](https://github.com/ggml-org/llama.cpp/discussions/5063)

### 1.7 Fused Q4K Matmul - ❌ SLOWER THAN DEQUANT

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Change `useFusedQ4K = true` | P0 | ⚠️ Disabled | Enabled but bypassed via kernel hints |
| Validate Q4K block alignment | P0 | ✅ Done | Row-wise/column-wise implemented |
| Benchmark fused vs separate | P0 | ✅ **DONE** | **Fused is 2.7x SLOWER** |

**Status:** ❌ **FUSED KERNEL IS SLOWER** - Benchmarks show dequant path is faster.

**Benchmark Results (Dec 2025):**

| Path | Layout | tok/s | Notes |
|------|--------|-------|-------|
| **Dequant → F16 GEMV** | column_wise | **8.0** | **DEFAULT** |
| Dequant → F16 GEMV | flat | 7.0 | Good fallback |
| Fused Q4K kernel | row_wise | 3.0 | **2.7x SLOWER** |

**Root Cause: Poor Thread Utilization**

The fused Q4K kernel (`matmul_q4_fused.wgsl`) has a fundamental design issue:

```
For Gemma 3 1B with K=1152:
- Q4K block size: 256 weights
- Blocks per row: ceil(1152/256) = 5 blocks
- Threads per workgroup: 256

Problem: 5 blocks ÷ 256 threads = 5 active threads
         251 of 256 threads (98%) are IDLE per workgroup!
```

**Why this happens:**
```wgsl
// matmul_q4_fused.wgsl (simplified)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let block_idx = lid.x;  // 0-255
  let num_blocks = (K + 255) / 256;  // = 5 for K=1152

  if (block_idx >= num_blocks) { return; }  // 251 threads exit immediately!

  // Only 5 threads do actual work...
}
```

**Current mitigation:**
- Converter defaults to `--q4k-layout column_wise`
- Loader uses dequant path via `kernelHints.q4kMatmul = 'dequant_f16'`
- Fused kernel still available for future optimization

**Future fix options:**
1. **Redesign kernel:** Multiple blocks per thread (loop over blocks)
2. **2D workgroup:** Use [32, 8, 1] instead of [256, 1, 1]
3. **Different kernel for small K:** Switch strategy based on K dimension

**Files:**
- `gpu/kernels/matmul_q4_fused.wgsl` - Fused kernel (needs redesign)
- `gpu/kernel-hints.ts` - `q4kMatmul: 'dequant_f16'` bypasses fused
- `loader/doppler-loader.ts` - `useFusedQ4K` flag (currently ignored via hints)

### 1.8 FFN Gate+Up Fusion

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Concatenate gate+up weights in writer | P0 | ✅ Done | `--fuse-gate-up` flag in convert-cli |
| Add `gate_up_proj` tensor support | P0 | ✅ Done | Loader and manifest support |
| Update FFN to use fused path | P0 | ✅ Done | `layer.ts` and `ffn.ts` handle gateUp |
| Add split+SiLU kernel | P1 | ✅ Done | `runSiLURowSplit` kernel |

**Status:** Implemented. Models converted with `--fuse-gate-up` use 2 matmul passes.
**Impact:** 1.2-1.3x FFN speedup (gate+up fused, then down)

```typescript
// Current: 3 passes
gate = matmul(input, gateWeight)   // Pass 1
up = matmul(input, upWeight)       // Pass 2
out = matmul(silu(gate)*up, down)  // Pass 3

// Proposed: 2 passes
gateUp = matmul(input, gateUpWeight)  // Pass 1 (fused)
out = matmul(silu_split(gateUp), down) // Pass 2
```

### 1.9 Workgroup Auto-Tuning (Complete)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Add benchmark loop to `_tuneAttention()` | P1 | ✅ Done | Benchmarks 1D workgroup sizes |
| Add benchmark loop to `_tuneSoftmax()` | P1 | ✅ Done | Benchmarks 1D workgroup sizes |
| Add benchmark loop to `_tuneRMSNorm()` | P1 | ✅ Done | Benchmarks 1D workgroup sizes |
| Add benchmark loop to `_tuneDequant()` | P1 | ✅ Done | Benchmarks 1D workgroup sizes |

**Current:** Matmul + attention/softmax/rmsnorm/dequant have benchmark loops.
**Fix:** Done; kernel-tuner now evaluates 1D candidates for these kernels.

### 1.10 Precision Optimization (Q4/BF16/F16)

**Target Precision Stack:**
```
Weights:     Q4_K_M (quantized, 4-bit) → keeps model size small
Matmul:      Fused Q4K kernel (dequant + matmul in one pass)
KV Cache:    F16 (not F32) → 2x memory savings
Activations: F16 where possible, F32 for numerically sensitive ops
Embeddings:  BF16 → F16 (converted at load time)
Norms:       BF16 → F32 (for numerical stability)
```

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Enable fused Q4K matmul | P0 | ✅ Done | `useFusedQ4K = true` in doppler-loader.ts |
| Default KV cache to F16 | P0 | ✅ Done | `init.ts` auto-detects F16 when GPU supports it |
| F16 matmul output for F16 weights | P0 | ✅ Done | matmul.ts selects F16 output when inputs are F16 |
| BF16→F16 for matmul weights | P0 | ✅ Done | `_shouldDequantizeToF16()` + spans path fixed |
| Remove unnecessary F32 intermediates | P1 | ⬜ TODO | Audit pipeline for F32 allocations |
| F16 activation pipeline | P2 | ⬜ TODO | Trade-off: speed vs accuracy |

**Implementation Notes:**

1. **KV Cache auto-detects F16:**
```typescript
// init.ts:267-269
const caps = getKernelCapabilities();
const kvDtype = caps?.hasF16 ? 'f16' : 'f32';
```

2. **Fused Q4K enabled:**
```typescript
// doppler-loader.ts - useFusedQ4K = true by default
```

3. **BF16 → F16 conversion:**
```typescript
// Both direct load and spans path now convert BF16 → F32 → F16 for matmul weights
// This enables optimized F16 GEMV kernels
```

**WebGPU Precision Constraints:**
- WebGPU has **no native BF16 support** - must convert to F16 or F32
- F16 requires `shader-f16` feature (detected via `gpuCaps.hasF16`)
- Q4K fused kernel requires subgroup support (detected via `gpuCaps.hasSubgroups`)

**Expected Impact:**
| Change | Memory Savings | Speed Impact |
|--------|---------------|--------------|
| F16 KV cache | 2x KV memory | ~same |
| Fused Q4K | ~same | 1.3-1.5x faster |
| F16 activations | 2x activation memory | ~1.2x faster (bandwidth) |

### 1.11 F16 GEMV Multi-Column Kernel (LM Head)

**Status:** ✅ Implemented, marginal impact

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Add `gemv_subgroup_multicol` config to utils.ts | P0 | ✅ Done | Lines 89-96 |
| Update runMatmul variant selection for N > 8192 | P0 | ✅ Done | Lines 263-267 |
| Update recordMatmul variant selection | P0 | ✅ Done | Lines 546-553 |
| Update workgroup calculation for 32 cols/wg | P0 | ✅ Done | Lines 314-315, 581-582 |
| Update dispatch logic for multicol | P0 | ✅ Done | Lines 404-406, 664-666 |
| Benchmark performance impact | P0 | ✅ Done | ~7 tok/s (no change from 4-col kernel) |

**Problem Identified:**

For Gemma 3's 262K vocab LM head with F16 tied embeddings:
- Original `gemv_subgroup`: 4 columns/workgroup → 65,536 workgroups
- New `gemv_subgroup_multicol`: 32 columns/workgroup → 8,192 workgroups (8x fewer)

```
LM head: M=1, N=262144, K=1152
Weight size: 262144 × 1152 × 2 bytes (F16) = 603MB per token read
```

**Implementation:**

```typescript
// matmul.ts selection logic (lines 263-267)
if (N > MULTICOL_THRESHOLD) {  // MULTICOL_THRESHOLD = 8192
  variant = 'gemv_subgroup_multicol';
} else {
  variant = 'gemv_subgroup';
}

// Workgroup dispatch (lines 314-315)
if (variant === 'gemv_subgroup_multicol') {
  gemvWorkgroupsX = Math.ceil(N / 32);  // 32 columns per workgroup
}
```

**Findings:**

| Metric | Before (4-col) | After (32-col) | Change |
|--------|----------------|----------------|--------|
| Workgroups | 65,536 | 8,192 | -87% |
| Decode tok/s | ~7 | ~7 | ~0% |
| Per-token latency | ~140ms | ~140ms | ~0% |

**Analysis:**

The 8x reduction in workgroups did NOT improve performance. This indicates:

1. **LM head is not the dominant bottleneck** - The 26 transformer layers have 4 matmuls each (Q/K/V/O projections) plus 2 FFN matmuls = 156 matmul operations per forward pass. The single LM head matmul may be <5% of total time.

2. **GPU timestamp profiling available** - Use `gpu/profiler.ts` to measure individual kernel execution times and identify the actual bottleneck.

3. **Memory bandwidth limited** - 603MB weight read per token at theoretical 200GB/s = 3ms minimum. Observed ~140ms suggests compute or other overheads dominate.

**Next Steps:**
- Profile with GPU timestamps to identify slowest kernels
- Consider speculative decoding to amortize LM head across multiple tokens
- Optimize layer matmul kernels if they prove to be bottlenecks

**Files Modified:**
- `gpu/kernels/utils.ts` - Added `gemv_subgroup_multicol` config
- `gpu/kernels/matmul.ts` - Updated selection and dispatch logic
- `gpu/kernels/matmul_gemv_subgroup.wgsl` - Kernel already existed (lines 130-213)

### 1.12 Readback Minimization (Critical)

**Current:** 128KB/token (full vocab logits: 262144 × 4 bytes)
**Target:** 4 bytes/token (single token ID)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| GPU argmax kernel | P0 | ✅ Done | Returns single u32 token ID |
| GPU top-k sampling | P0 | ✅ Done | Samples on GPU, reads only token ID |
| Measure readback overhead | P0 | ⬜ TODO | Isolate GPU→CPU transfer time |

**Impact:** Each 128KB readback costs 2-6ms. At 8 tokens, that's 1MB total readback.

**Measurement:**
```bash
# Check readback bytes per run
npm run doppler -- bench inference --prompt xs 2>&1 | grep "readback"
# Should see: gpu_readback_bytes_total in results JSON
```

**Implementation options:**
1. **GPU argmax:** Single parallel reduction → read 1 u32
2. **GPU top-k + sample:** Full sampling on GPU → read 1 u32
3. **Streaming readback:** Read only top-k logits (~1KB) instead of full vocab

### 1.13 Kernel Utilization Audit (P0)

**Goal:** Ensure all compute kernels have >50% thread utilization.

| Kernel | Workgroup | Active Threads | Utilization | Status |
|--------|-----------|----------------|-------------|--------|
| `q4_fused` | 256 | ceil(K/256) | **2%** (K=1152) | ⚠️ Deprecated (use multicol) |
| `q4_fused_multicol` | 256 | 32×8 | **100%** | ✅ Fixed (always used for M=1) |
| `q4_fused_batched` | 64×4 | 64×M | Varies | ⚠️ Audit |
| `gemv_subgroup` | 256 | 256 | **100%** | ✅ |
| `gemv_subgroup_multicol` | 256 | 256 | **100%** | ✅ |
| `dequant_q4k` | 64 | N×K/256 | **100%** | ✅ |
| `attention_*` | Varies | Varies | ⚠️ Audit | TODO |
| `rmsnorm` | 256 | hidden_size | **100%** | ✅ |
| `silu` | 256 | N | **100%** | ✅ |

**Fix Applied (Dec 2025):**
```typescript
// matmul.ts - Lowered threshold to use multicol for ALL layer matmuls
const MULTICOL_THRESHOLD = 256;  // Was 8192
// Now q4_fused_multicol used for q_proj (N=1024), gate_proj (N=6912), etc.
```

**Audit Checklist:**
- [x] Q4K fused GEMV (fixed: always use multicol for M=1, 100% utilization)
- [ ] Q4K fused batched (M>1 prefill)
- [ ] Attention kernels (tiled_large, tiled_small, streaming)
- [ ] RoPE kernel
- [ ] Gather/embedding kernel

**How to audit a kernel:**
```
1. Find workgroup size: @compute @workgroup_size(X, Y, Z)
2. Count threads that exit early: if (id >= limit) { return; }
3. Calculate: utilization = active_threads / (X × Y × Z)
4. Fix if utilization < 50%
```

### 1.14 Kernel Fusion Opportunities (Comprehensive)

**Goal:** Reduce kernel launch overhead and memory bandwidth by fusing operations.

#### Currently Implemented Fusions ✅

| Fusion | Passes | Speedup | Status | Files |
|--------|--------|---------|--------|-------|
| Gate+Up FFN | 3→2 | 1.2-1.3x | ✅ Done | `ffn.ts`, `rdrr-writer.ts` |
| FlashAttention (tiled + online softmax) | N→1 | 2x | ✅ Done | `attention.wgsl` |
| Logits+Argmax+Sampling | 3→1 | 1.3-1.5x | ✅ Done | `logits.ts`, `sample.wgsl` |
| Dequant Q4K → F16 GEMV | 2 | 2.3x | ✅ Active | `dequant_subgroup.wgsl`, `matmul_gemv_subgroup.wgsl` |
| Multi-column Q4K GEMV | - | 8x fewer wg | ✅ Done | `matmul_q4_fused.wgsl:main_multicol` |
| Subgroup GEMV | - | 1.5x | ✅ Done | `matmul_gemv_subgroup.wgsl` |
| Command buffer batching | 260→1 submits | 50-100ms | ✅ Done | `command-recorder.ts` |

#### Missing Fusion Opportunities (Priority Order)

##### P0 - High Impact, Should Implement

| Fusion | Current | Proposed | Est. Speedup | Complexity | Status | Notes |
|--------|---------|----------|--------------|------------|--------|-------|
| **Full F16 Activation Pipeline** | F32 activations | F16 throughout | 1.5-2x | High | ⬜ TODO | WebLLM uses q4f16 (24ms/tok) vs DOPPLER q4f32 (125ms/tok). Blocked on numerical stability verification. |
| **Residual+RMSNorm** | 2 passes | 1 pass | 1.2-1.3x | Low | ✅ **Done** | **Actual: ~1.5% impact** - RMSNorm is <1% of GPU time (~0.42ms), attention dominates at 86% (~73ms). Kernel optimized with shared memory cache to avoid double-load. Files: `layer.ts:467-493,634-701`, `rmsnorm.wgsl:207-263` |
| **Quantized Matmul+RMSNorm** | 2-3 passes | 1 pass | 1.2-1.5x | Medium | ⬜ TODO | Fuse dequant+matmul+norm. Requires custom shader. |

##### P1 - Medium Impact

| Fusion | Current | Proposed | Est. Speedup | Complexity | Notes |
|--------|---------|----------|--------------|------------|-------|
| **Matmul+RMSNorm Epilogue** | 2 passes | 1 pass | 1.1-1.3x | Medium | Compute norm while accumulating matmul output. Harder with current architecture since RMSNorm needs full output. |
| **Attention+Residual** | 2 passes | 1 pass | 1.1-1.2x | Medium | Fuse residual add into attention kernel epilogue. Requires attention kernel variant. |

##### P2 - Lower Priority / More Complex

| Fusion | Current | Proposed | Est. Speedup | Complexity | Notes |
|--------|---------|----------|--------------|------------|-------|
| **Matmul+SiLU Epilogue** | 2 passes | ~1.5 passes | 1.1-1.2x | Medium | Fuse first FFN matmul with split+SiLU. Split output at epilogue, apply SiLU to gate half. |
| **Parallel FFN Gate+Up+Down** | Sequential | Concurrent | 1.1-1.3x | Medium | Record gate/up matmuls in parallel (independent ops). Needs `recordParallelGroup()`. |
| **Parallel Q/K/V Projection** | Sequential | Concurrent | 1.1-1.2x | Medium | Record Q/K/V projections in parallel before sync. Same as FFN above. |

#### Fusion Implementation Details

##### Residual+RMSNorm Fusion (P0, Low Complexity) ✅ **DONE Dec 2025**

**Status:** ✅ Implemented - Actual impact ~1.5% (minimal because RMSNorm <1% of GPU time)

**Current flow (before):**
```typescript
// 2 separate kernel passes
let postAttn = await doRMSNorm(attnOutput, normWeight, eps, {...});
postAttn = await doResidualAdd(postAttn, inputBuffer, {...});
```

**Implemented fused flow:**
```typescript
// 1 kernel pass with residual parameter
const postAttn = await doRMSNorm(attnOutput, normWeight, eps, {
  batchSize: numTokens,
  hiddenSize,
  residual: inputBuffer,  // FUSION: Add residual in same kernel
}, recorder);
```

**Fused kernel implementation:**
```wgsl
// rmsnorm.wgsl:207-263 - rmsnorm_inplace_residual with shared memory cache
@compute @workgroup_size(256, 1, 1)
fn rmsnorm_inplace_residual(...) {
    // First pass: compute (input + residual), cache it, compute sum of squares
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < size) {
            // OPTIMIZATION: Compute residual add once, cache in shared memory
            let x = input[baseOffset + idx] + residual[baseOffset + idx];
            shared_cache[idx] = x;  // Cache for second pass
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    // Reduction to compute RMS...

    // Second pass: use cached values (no duplicate loads!)
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < size) {
            let x = shared_cache[idx];  // Read from cache
            output[baseOffset + idx] = x * inv_rms * weight[idx];
        }
    }
}
```

**Files modified:**
- `gpu/kernels/rmsnorm.wgsl:207-263` - Optimized fused variant with shared memory cache
- `gpu/kernels/rmsnorm.ts:17,149` - Already had `residual` parameter support
- `inference/pipeline/layer.ts:467-493` - Fused attention path (Gemma)
- `inference/pipeline/layer.ts:634-701` - Fused FFN path (Gemma)

**Performance Results (Dec 19, 2025):**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode tok/s | 5 | 5 | ~0% |
| Latency P50 | 201ms | 199ms | -1% |
| RMSNorm GPU time | ~0.4ms | ~0.42ms | +5% variance |
| Attention GPU time | 67ms | 73ms | +9% variance |

**Key Finding:** Minimal impact because RMSNorm is **<1% of GPU time**. The real bottleneck is attention (86% of GPU time at 67-73ms/token). See section 1.22 for attention optimization plan.

##### Full F16 Activation Pipeline (P0, High Impact)

**Current:** F16 weights + F16 KV cache, but **F32 activations throughout**
**Target:** F16 activations everywhere except numerically sensitive ops

**Implementation steps:**
1. Audit all intermediate buffers in `inference/pipeline/`
2. Change allocation dtype from `f32` to `f16` for:
   - Attention Q/K/V projections
   - FFN gate/up/down outputs
   - Layer hidden states
3. Keep F32 for:
   - RMSNorm internal computation (divide)
   - Softmax accumulator
   - Final logits
4. Verify numerical stability with reference comparison

**Expected impact:**
- 2x memory bandwidth reduction
- 1.5-2x speedup (matches WebLLM's q4f16 mode)

**Risk:** Numerical underflow in deep layers. Need per-layer magnitude verification.

##### Parallel Kernel Execution (P2)

**Current:** Sequential kernel recording
```typescript
recordMatmul(encoder, Q_proj);
recordMatmul(encoder, K_proj);
recordMatmul(encoder, V_proj);
```

**Proposed:** Add `recordParallelGroup()` to CommandRecorder
```typescript
recorder.recordParallelGroup([
    () => recordMatmul(encoder, Q_proj),
    () => recordMatmul(encoder, K_proj),
    () => recordMatmul(encoder, V_proj),
]);
```

**Implementation:**
- `gpu/command-recorder.ts` - Add `recordParallelGroup()` method
- Group kernels share same command encoder pass
- WebGPU can overlap execution of independent compute dispatches

**Files to modify:**
- `gpu/command-recorder.ts` - New method
- `inference/pipeline/attention.ts` - Parallel Q/K/V
- `inference/pipeline/ffn.ts` - Parallel gate/up

### 1.15 Speedup Optimizations (Comprehensive)

**Goal:** Maximize throughput through non-fusion optimizations.

#### Currently Implemented ✅

| Optimization | Speedup | Status | Files |
|--------------|---------|--------|-------|
| Column-wise Q4K layout | 2.7x | ✅ Default | `convert-cli.ts`, `quantizer.ts` |
| F16 KV cache auto-detect | - | ✅ Done | `init.ts` |
| BF16→F16 matmul weights | 1.2-1.5x | ✅ Done | `dequant.ts` |
| GPU sampling (no readback) | 1.3-1.5x | ✅ Done | `sample.ts`, `logits.ts` |
| Shader prewarm during load | - | ✅ Done | `pipeline.ts` |
| Buffer reuse strategy | - | ✅ Done | `buffer-pool.ts` |

#### Missing Speedup Opportunities

##### P0 - Critical Path

| Optimization | Est. Speedup | Complexity | Status | Notes |
|--------------|--------------|------------|--------|-------|
| **GPU Timestamp Profiling** | - (diagnostic) | Low | ✅ Done | Benchmarks now populate `gpu_time_ms_prefill/decode` via `gpu/profiler.ts`. |
| **Fix Fused Q4K Thread Utilization** | 2.7x (recover) | Medium | ✅ Done | Fixed by always using multicol variant for M=1, giving 100% utilization (32 cols × 8 threads). |

##### P1 - Significant Impact

| Optimization | Est. Speedup | Complexity | Status | Notes |
|--------------|--------------|------------|--------|-------|
| **Complete Workgroup Auto-Tuning** | 1.1-1.2x | Medium | ✅ Done | Benchmarks now cover attention, softmax, rmsnorm, dequant; rope/silu still hardcoded. |
| **Shader Constants → Uniforms Migration** | Config fidelity | Medium | ⬜ TODO | Move baked constants to uniforms for runtime tuning. See 1.16. |
| **Remove F32 Intermediate Allocations** | 1.1-1.2x | Medium | ⬜ TODO | Audit pipeline for unnecessary F32 buffers. |

##### P2 - Future Optimizations

| Optimization | Est. Speedup | Complexity | Status | Notes |
|--------------|--------------|------------|--------|-------|
| **Speculative Decoding** | 2-3x | High | ⬜ Framework ready | Needs draft model wiring. Amortize LM head across multiple tokens. |
| **Tensor Parallelism** | 2x (2 GPU) | High | ⬜ TODO | Split large matmuls across GPUs. WebGPU multi-adapter support needed. |
| **Continuous Batching** | N/A (throughput) | Medium | ⬜ TODO | Handle multiple concurrent requests. |

#### Workgroup Auto-Tuning Completion (P1)

**Current state:** Matmul + attention/softmax/rmsnorm/dequant have benchmark loops in `kernel-tuner.ts`

| Kernel | Current | Has Benchmark | TODO |
|--------|---------|---------------|------|
| matmul | Auto-tuned | ✅ Yes | - |
| attention | Benchmarked | ✅ Yes | - |
| softmax | Benchmarked | ✅ Yes | - |
| rmsnorm | Benchmarked | ✅ Yes | - |
| dequant | Benchmarked | ✅ Yes | - |
| rope | `[256, 1, 1]` hardcoded | ❌ No | Add `_tuneRoPE()` |
| silu | `[256, 1, 1]` hardcoded | ❌ No | Add `_tuneSiLU()` |

**Implementation:** Copy `_tuneMatmul()` pattern for each kernel.

#### Fix Fused Q4K Kernel (P0) ✅ **DONE Dec 2025**

**Status:** ✅ Fixed - Always use multicol variant for M=1 decode

**Problem (before fix):** 98% thread idle for small K dimensions

```
K=1152: ceil(1152/256) = 5 blocks per row
Workgroup (q4_fused): 256 threads
Active: 5 threads (2% utilization) ❌
```

**Solution Implemented:**

Changed kernel selection in `matmul.ts:254-255` to always use `q4_fused_multicol` for M=1:

```typescript
// Before:
if (M === 1) {
  variant = N > MULTICOL_THRESHOLD ? 'q4_fused_multicol' : 'q4_fused';
}

// After:
if (M === 1) {
  variant = 'q4_fused_multicol';  // Always multicol for GEMV
}
```

**Result:**
```
Workgroup (q4_fused_multicol): 256 threads
Columns per workgroup: 32
Threads per column: 8
Active: 32 × 8 = 256 threads (100% utilization) ✅
```

**Files Modified:**
- `gpu/kernels/matmul.ts:254-255` - Always use multicol for M=1

**Alternative Options Considered (not implemented):**

1. **Loop over blocks** - More complex, requires kernel rewrite
2. **2D workgroup layout** - Dispatcher complexity
3. **K-threshold switching** - Already using dequant path by default (2.3x faster)

### 1.16 Shader Configuration Audit (P1)

**Goal:** Use uniforms over constants for runtime configurability.

**Why this matters:**
- `const` values are compiled into shader → requires shader recompilation to change
- `uniform` values are set at dispatch time → can be configured via manifest/kernelHints
- Enables manifest → config → kernel layering without shader rebuilds

**Audit Checklist:**
| Kernel | Hardcoded Constants | Should Be Uniform | Status |
|--------|---------------------|-------------------|--------|
| `matmul_q4_fused.wgsl` | `COLS_PER_WG=32`, `THREADS_PER_COL_GEMV=8` | Yes (tune per device) | ⬜ TODO |
| `matmul_gemv_subgroup.wgsl` | `MULTICOL=32` | Yes | ⬜ TODO |
| `attention_*.wgsl` | `TILE_SIZE`, `HEAD_DIM` | Partial (head_dim varies) | ⬜ TODO |
| `dequant.wgsl` | `BLOCK_SIZE=256` | No (Q4K spec) | ✅ OK |
| `rmsnorm.wgsl` | `WG_SIZE=256` | Maybe | ⬜ TODO |

**Example migration:**
```wgsl
// Before (hardcoded):
const COLS_PER_WG: u32 = 32u;

// After (configurable via uniform):
struct KernelConfig {
    cols_per_wg: u32,
    threads_per_col: u32,
    // ... other tuning params
}
@group(0) @binding(4) var<uniform> config: KernelConfig;
```

**Benefits:**
1. Manifest `kernelHints.colsPerWorkgroup` → config struct → shader uniform
2. Device-specific tuning without shader variants
3. Auto-tuner can test different configs without recompilation

### 1.17 Complete Optimization Opportunity Matrix

**All optimizations ranked by impact and implementation complexity.**

#### Implemented ✅ (Baseline: 4 tok/s ⚠️ - was 8 tok/s, investigating regression)

| # | Optimization | Speedup | Cumulative | Status |
|---|--------------|---------|------------|--------|
| 1 | Column-wise Q4K layout | 2.7x | 2.7x | ✅ Done |
| 2 | FlashAttention fusion | 2x | 5.4x | ✅ Done |
| 3 | Subgroup GEMV | 1.5x | 8.1x | ✅ Done |
| 4 | GPU sampling (no readback) | 1.3-1.5x | 10-12x | ✅ Done |
| 5 | Gate+Up FFN fusion | 1.2-1.3x | 12-16x | ✅ Done |
| 6 | Command buffer batching | - | - | ✅ Done |
| 7 | F16 KV cache | - | - | ✅ Done |
| 8 | BF16→F16 weights | 1.2-1.5x | - | ✅ Done |

#### TODO - Ordered by Priority × Impact

| # | Optimization | Est. Speedup | Priority | Complexity | Status |
|---|--------------|--------------|----------|------------|--------|
| 0 | **🔥 Fix GPU Submit Overhead** | 1.5x | P0 | Medium | ⬜ 6.3→1-2 submits/tok |
| 1 | **🔥 Fix Dequant/GEMV Regression** | 1.3x | P0 | Medium | ⬜ Investigate root cause |
| 2 | **Full F16 Activation Pipeline** | 1.5-2x | P0 | High | ⬜ Numerical stability verification |
| 3 | **GPU Timestamp Profiling** | - (diagnostic) | P0 | Low | ✅ Done |
| 4 | **Residual+RMSNorm Fusion** | ~~1.2-1.3x~~ **~1.5%** | ~~P0~~ | Low | ✅ **Done** (minimal impact) |
| 5 | **Quantized Matmul+RMSNorm** | 1.2-1.5x | P0 | Medium | ⬜ Custom shader |
| 6 | **Fix Fused Q4K Utilization** | 2.7x (recover) | P0 | Medium | ✅ **Done** (multicol variant) |
| 7 | **Complete Workgroup Auto-Tuning** | 1.1-1.2x | P1 | Medium | None |
| 8 | **Shader Constants → Uniforms** | Config | P1 | Medium | None |
| 9 | **Matmul+RMSNorm Epilogue** | 1.1-1.3x | P1 | Medium | RMSNorm needs full output |
| 10 | **Attention+Residual Fusion** | 1.1-1.2x | P1 | Medium | Attention kernel variant |
| 11 | **Remove F32 Intermediates** | 1.1-1.2x | P1 | Medium | Audit needed |
| 12 | **Attention Decode Kernel (GEMV-style)** | 4-8x attention | P1 | High | ✅ **Done** (subgroup, 80→4 barriers) |
| 13 | **Matmul+SiLU Epilogue** | 1.1-1.2x | P2 | Medium | Custom shader |
| 14 | **Parallel FFN Gate+Up+Down** | 1.1-1.3x | P2 | Medium | `recordParallelGroup()` |
| 15 | **Parallel Q/K/V Projection** | 1.1-1.2x | P2 | Medium | `recordParallelGroup()` |
| 16 | **GPU-Only Decode Loop (N tokens)** | 2-5x | P1 | Medium | ⏳ **Partial** (infra done, needs integration) |
| 17 | **Speculative Decoding** | 2-3x | P2 | High | Draft model wiring |
| 18 | **Tensor Parallelism** | 2x (2 GPU) | P2 | High | WebGPU multi-adapter |
| 19 | **Continuous Batching** | N/A | P2 | Medium | None |

#### Theoretical Maximum Speedup

**Step 1: Fix regression (P0) - recover 2x**
```
Current: 4 tok/s (265ms/tok) ⚠️ REGRESSED
+ Fix GPU submit overhead (6.3 → 1-2/tok): ~1.5x → 6 tok/s
+ Fix dequant/GEMV regression: ~1.3x → 8 tok/s (baseline recovered)
```

**Step 2: Apply optimizations (P0+P1) - compound**
```
Baseline: 8 tok/s (125ms/tok)
+ F16 activations: 1.75x → 14 tok/s (71ms/tok)
+ Residual+RMSNorm: 1.25x → 17.5 tok/s (57ms/tok)
+ Quantized Matmul+RMSNorm: 1.35x → 24 tok/s (42ms/tok)
+ Workgroup tuning: 1.15x → 27 tok/s (37ms/tok)
+ Remove F32 intermediates: 1.15x → 31 tok/s (32ms/tok)
= Target: ~30-35 tok/s (29-33ms/tok)

With P2 speculative decoding: 2.5x → 75-85 tok/s
```

**WebLLM comparison:** 41 tok/s (24ms/tok)
**Target parity:** 40+ tok/s

### 1.18 WebLLM Comparison Testing (P0)

**Goal:** Apples-to-apples comparison on same models WebLLM benchmarks.

#### WebLLM Published Benchmarks (M3 Max)

| Model | WebLLM | MLC-LLM Native | Retained |
|-------|--------|----------------|----------|
| Llama-3.1-8B (Q4) | 41.1 tok/s | 57.7 tok/s | 71% |
| Phi-3.5-mini 3.8B (Q4) | 71.1 tok/s | 89.3 tok/s | 80% |

WebLLM also supports Qwen2 0.5B/1.5B/7B but has no published benchmarks.

#### DOPPLER Validation Matrix

| Model | Size | Architecture | Priority | Status | Notes |
|-------|------|--------------|----------|--------|-------|
| **Phi-3.5-mini-instruct** | 3.8B | Standard | P0 | ⬜ TODO | WebLLM's best case (71 tok/s) |
| **Qwen2.5-1.5B-Instruct** | 1.5B | Standard | P0 | ⬜ TODO | Similar to Gemma 1B |
| **Llama-3.2-1B-Instruct** | 1B | Standard | P0 | ⬜ TODO | Baseline Llama arch |
| **Llama-3.2-3B-Instruct** | 3B | Standard | P1 | ⬜ TODO | Mid-size comparison |
| Llama-3.1-8B-Instruct | 8B | Standard | P2 | ⬜ TODO | Requires 16GB+ VRAM |

#### Why This Matters

- Gemma 3 has unusual architecture (sandwich norms, sliding window pattern) — bugs may be model-specific
- Testing Llama/Qwen/Phi validates pipeline on "standard" transformer arch
- Direct tok/s comparison on same M3 hardware settles the debate

#### Test Procedure

1. Convert model: `doppler convert <hf-model> output/ --quantize q4_k_m`
2. Run benchmark: `doppler bench inference --prompt xs --headed`
3. Compare metrics: decode tok/s, prefill tok/s, TTFT, VRAM

#### Expected Outcome

If DOPPLER achieves similar % of WebLLM on Phi-3.5 as Gemma:
- WebLLM Phi-3.5: 71 tok/s
- DOPPLER target: ~28-57 tok/s (40-80% retained)
- Current Gemma: 4 tok/s → expect ~4-8 tok/s on Phi if issue is systemic

### 1.19 WebLLM vs DOPPLER: Claims vs Evidence

**Goal:** Document where DOPPLER challenges or contradicts WebLLM assumptions with citations.

#### Vision (Design Divergence from WebLLM)

| WebLLM Claim | DOPPLER Challenge | Evidence |
|--------------|-------------------|----------|
| AOT-compiled kernels required | Runtime kernel hints + overrides | `storage/rdrr-format.ts`, `inference/pipeline.ts`, `tools/doppler-cli.ts` |
| TVM compiler generates FlashAttention | Manual WGSL FlashAttention-style fusion | `docs/EXECUTION_PIPELINE.md`, `gpu/kernels/attention.wgsl` |
| WASM library per model | RDRR shards + manifest (no WASM) | `loader/doppler-loader.ts`, `storage/rdrr-format.ts` |

#### Reality (Measured Results Contradicting WebLLM Assumptions)

| WebLLM Assumption | DOPPLER Evidence | Citation |
|-------------------|------------------|----------|
| **Fusion is always a win** | Fused Q4K has 13x worse TTFT (63s vs 4.8s) while matching decode speed | `tests/results/pipeline_gemma-1b-q4-row_*_2025-12-20T01-46-14-701Z.json` vs `*_2025-12-20T00-23-00-909Z.json` |
| **1-2 submits per forward pass** | Real benchmark: 396 decode submits for 100 tokens (~4 submits/token) | `docs/EXECUTION_PIPELINE.md`, `tests/results/bench_gemma-1b-q4-col_xs_default.json` |
| **JS overhead negligible (~0.5ms)** | Real decode ~1 tok/s on Q4 row (long prompts) — not GPU-bound | `docs/ARCHITECTURE.md`, `tests/results/pipeline_gemma-1b-q4-row_*_2025-12-20T00-23-00-909Z.json` |
| **80% native retention** | DOPPLER: 4 tok/s vs WebLLM 71 tok/s = 5.6% retention | `tests/results/bench_gemma-1b-q4-col_xs_default.json` |

#### Key Findings

1. **Fusion can regress performance** — Fused Q4K kernel has 98% thread idle (5/256 active for K=1152). Dequant→GEMV path is 2.7x faster.

2. **Command batching not working as designed** — Design claims 1-2 submits, reality is 4-6.3 submits/token. Direct `queue.submit()` bypasses in:
   - `inference/pipeline/layer.ts`
   - `inference/pipeline/attention.ts`
   - `inference/pipeline/logits.ts`
   - `inference/pipeline/embed.ts`
   - `inference/kv-cache.ts`

3. **JS orchestration overhead dominates** — Architecture doc claims ~0.5ms JS vs ~25ms GPU, but actual decode latency (265ms/tok) suggests orchestration, not compute, is the bottleneck.

#### DOPPLER's Thesis

DOPPLER bets that:
- Hand-written WGSL kernels can match compiler-generated code
- Runtime flexibility (manifest hints) beats AOT rigidity
- No TVM/MLC-LLM dependency = simpler deployment

**Current status:** The thesis is not yet validated. 10x performance gap (4 vs 40+ tok/s) must close before claims can be substantiated.

### 1.20 GPU-Only Decode Loop (P1) — N Tokens Without CPU Roundtrip ⏳ **Partial**

**Status:** ✅ Infrastructure complete, needs integration & testing

**Goal:** Generate 5-100 tokens on GPU without reading back to CPU between tokens.

#### Current Flow (1 readback per token)

```
Token 1: GPU forward → submit → readback → JS sampling → GPU forward
Token 2: GPU forward → submit → readback → JS sampling → GPU forward
...
Token N: GPU forward → submit → readback → JS sampling → done

Total: N submits, N readbacks, N JS→GPU roundtrips
At 265ms/tok with 6.3 submits/tok = ~42ms per submit overhead
```

#### Proposed Flow (1 readback per N tokens)

```
GPU Command Buffer:
  for i in 0..N:
    forward_pass(token[i])      // Layers + logits
    sample_token()              // Argmax/top-k on GPU
    check_stop_condition()      // EOS/max_tokens on GPU
    if stop: break
    embed_next_token()          // Gather for next iteration
    append_kv_cache()           // KV cache update

  readback(tokens[0..N])        // Single readback at end

Total: 1-2 submits, 1 readback, minimal JS overhead
```

#### What We Already Have ✅

| Component | Status | File |
|-----------|--------|------|
| GPU argmax sampling | ✅ Done | `gpu/kernels/sample.wgsl` |
| GPU top-k sampling | ✅ Done | `gpu/kernels/sample.wgsl` |
| **Fused decode with top-k** | ✅ Done Dec 20 | `inference/pipeline.ts:740-783` |
| `recordGPUSample()` batched sampling | ✅ Done Dec 20 | `gpu/kernels/sample.ts:409-496` |
| Token embedding lookup | ✅ Done | `gpu/kernels/gather.wgsl` |
| Command recorder | ✅ Done | `gpu/command-recorder.ts` |
| KV cache management | ✅ Done | `inference/kv-cache.ts` |

#### What We Need to Build

| Component | Complexity | Status | Description |
|-----------|------------|--------|-------------|
| **Stop condition kernel** | Low | ✅ Done | Check if sampled token == EOS or pos >= max_tokens |
| **GPU token buffer** | Low | ✅ Done | Buffer to store N sampled tokens on GPU |
| **Loop orchestration** | Medium | ✅ Done | Record N decode iterations in single command buffer (Option A) |
| **Early exit handling** | Medium | ✅ Done | CPU-side trim after readback (Option A) |
| **Streaming callback** | Medium | ⬜ TODO | Optional: periodic readback every K tokens for UI |
| **Embedding scaling** | Low | ✅ Done | Gemma embedding scale in batched path |
| **Integration & testing** | Medium | ✅ Done | Wired into `inference/pipeline.ts` generate loop |

#### Stop Condition Kernel (New)

```wgsl
@group(0) @binding(0) var<storage, read> sampled_token: u32;
@group(0) @binding(1) var<storage, read> eos_token: u32;
@group(0) @binding(2) var<storage, read> current_pos: u32;
@group(0) @binding(3) var<storage, read> max_tokens: u32;
@group(0) @binding(4) var<storage, read_write> should_stop: u32;

@compute @workgroup_size(1)
fn check_stop() {
    if (sampled_token == eos_token || current_pos >= max_tokens) {
        should_stop = 1u;
    }
}
```

#### Implementation Options

**Option A: Fixed N tokens (simplest)**
- Record exactly N decode steps
- Readback all N tokens at end
- Mask invalid tokens if EOS hit early
- Pro: Simple, no GPU branching
- Con: Wasted compute if EOS early

**Option B: GPU-side early exit (ideal)**
- Use indirect dispatch to conditionally skip remaining steps
- Requires `dispatchWorkgroupsIndirect` with condition buffer
- Pro: No wasted compute
- Con: More complex, needs careful buffer management

**Option C: Chunked decode (balanced)**
- Generate K tokens (e.g., 10) per submit
- Check stop condition between chunks
- Pro: Balances latency and efficiency
- Con: Still has some CPU roundtrips

#### Expected Speedup

| Current | With GPU Loop (N=10) | With GPU Loop (N=100) |
|---------|---------------------|----------------------|
| 6.3 submits/tok | 0.63 submits/tok | 0.063 submits/tok |
| 265ms/tok | ~50-80ms/tok | ~30-50ms/tok |
| 4 tok/s | 12-20 tok/s | 20-33 tok/s |

**Theoretical max:** If submit overhead is 42ms/tok and we eliminate 90% of submits, that's 38ms savings → 2-5x speedup.

#### Files Modified

| File | Change | Status |
|------|--------|--------|
| `gpu/kernels/check-stop.wgsl` | ✅ Created | Stop condition kernel (checks EOS/max_tokens) |
| `gpu/kernels/check-stop.ts` | ✅ Created | TypeScript wrapper with `recordCheckStop()` |
| `inference/pipeline.ts` | ✅ Modified | Added `_generateNTokensGPU()` private method (Option A) |
| `inference/pipeline.ts` | ✅ Done | Integrated batched generate loop |

#### Implementation Details

**Current Implementation:** Option A (Fixed N tokens, simplest)
- Records exactly N decode steps in single command buffer
- Each iteration: gather → layers → logits → argmax → check_stop
- Readback all N tokens + stop flags at end
- CPU post-process to find first EOS and trim
- **Limitation:** None; embedding scaling now supported in batched path via `recordScale`.

**Files:**
- `gpu/kernels/check-stop.wgsl` - WGSL kernel for stop condition check
- `gpu/kernels/check-stop.ts` - TypeScript wrapper for recording stop checks
- `inference/pipeline.ts` line 898-1089 - `_generateNTokensGPU()` implementation

**Architecture:**
```typescript
// Skeleton flow in _generateNTokensGPU()
for (let i = 0; i < N; i++) {
  hiddenStates = recordGather(tokenBuffers[i], ...)  // Read token[i]
  for (let l = 0; l < numLayers; l++) {
    hiddenStates = processLayer(l, hiddenStates, ...)  // Process through layer
  }
  logitsBuffer = recordLogitsGPU(hiddenStates, ...)  // Compute logits
  sampledToken = recordArgmax(logitsBuffer, ...)     // Sample token[i+1]
  copyBufferToBuffer(sampledToken → tokenBuffers[i+1])  // Store for next iter
  stopFlag = recordCheckStop(tokenBuffers[i+1], ...)    // Check if should stop
  copyBufferToBuffer(stopFlag → stopBuffer[i])       // Store stop flag
}
recorder.submit()  // Single submit for all N iterations!
```

**Next Steps:**
1. Add `recordScale` to `inference/pipeline/embed.ts` for Gemma batched scaling
2. Wire `_generateNTokensGPU()` into public generate API
3. Add `useGPULoop: boolean` option to GenerateOptions
4. Benchmark: measure actual speedup with N=10, N=50, N=100

### 1.21 Submit Path Investigation (P0)

**Goal:** Reduce GPU submits from 6.3/token to 1-2/token.

#### Submit Path Architecture

| WebLLM Claim | DOPPLER Module | Gap |
|--------------|----------------|-----|
| AOT-compiled kernels + graph fusion | `gpu/kernels/*.ts` + `storage/rdrr-format.ts` hints | No graph-level fusion |
| FlashAttention-style fused attention | `gpu/kernels/attention*.wgsl` | ✅ Have tiled+online softmax |
| WASM for CPU hot paths | `inference/pipeline/*.ts` (pure JS/TS) | JS overhead |
| Paged KV cache | `inference/kv-cache.ts` | ✅ Have, not paged |
| Compiled WASM library per model | `loader/doppler-loader.ts` + RDRR | Runtime flexibility |

#### Submit Path Analysis

**Batching primitive:** `gpu/command-recorder.ts`
- Created in `inference/pipeline.ts` for prefill/decode
- Single submit per pass when used correctly

**Submit tracking:** `gpu/submit-tracker.ts`
- Wraps `queue.submit()` calls in `gpu/device.ts`

**Direct submit bypasses (inflation sources):**
- `inference/pipeline/attention.ts` - direct submits when recorder is not used
- `inference/pipeline/logits.ts` - prefill now recorded; fallback still submits
- `inference/pipeline/embed.ts` - recorder path added; fallback still submits
- `inference/kv-cache.ts` - direct submits for non-recorded update paths

**Fused decode issue:**
- `pipeline.ts` uses `recordLogitsGPU` + `recordArgmax`
- Still issues extra submit for argmax staging buffer copy per token

#### Investigation Tasks

| Task | Priority | Status | Files |
|------|----------|--------|-------|
| Instrument submit counts by phase | P0 | ✅ Done | `gpu/submit-tracker.ts`, `gpu/device.ts` |
| Add recorder support to embed path | P0 | ✅ Done | `inference/pipeline/embed.ts` |
| Add recorder support to logits path | P0 | ✅ Done | `inference/pipeline.ts` (prefill) |
| Add recorder support to KV cache | P0 | ✅ Done | `inference/kv-cache.ts` |
| Audit unconditional readbacks | P0 | ✅ Done | `inference/pipeline/logits.ts`, `inference/pipeline/embed.ts` |
| Gate debug readbacks | P1 | ✅ Done | `window.DOPPLER_DEBUG_LOGITS`, `window.DOPPLER_DEBUG_KERNELS` |

#### Target

- Current: 6.3 submits/token
- Target: 1-2 submits/token
- Expected speedup: 1.5-2x from submit reduction alone

### 1.22 Attention Kernel Decode Inefficiency (P1) ✅ **Implemented**

**Status:** ✅ Subgroup decode kernel implemented (Dec 20, 2025) - ready for testing

#### Problem Analysis

The `attention_small_f16kv.wgsl` kernel has several inefficiencies for decode (seqLen=1):

| Issue | Impact | Details |
|-------|--------|---------|
| **Excessive barriers** | 80 barriers/token | For headDim=256: 8 head tiles (256/32) × 2 barriers × 5 KV blocks = 80 barriers |
| **Poor thread utilization** | 97% idle | Workgroup size=32, but seqLen=1 → 31 of 32 threads idle |
| **Nested loops** | Serialized compute | Lines 107-118 and 166-181 have nested loops that could be vectorized |

#### Why tiled_large Won't Work

```wgsl
// attention_f16kv.wgsl line 82
var acc: array<f32, 64>;  // Limited to headDim ≤ 64
```

Gemma 1B has headDim=256, so `tiled_large` variant is incompatible. This is a fundamental architectural limitation.

#### Barrier Calculation

```
headDim = 256
HEAD_TILE = 32
head_tiles = headDim / HEAD_TILE = 8

kvLen ≈ 135 (after prefill)
KV_BLOCK = 32
kv_blocks = ceil(135/32) = 5

barriers_per_token = kv_blocks × head_tiles × 2 = 5 × 8 × 2 = 80
```

Each `workgroupBarrier()` costs ~1-5μs on M3 → 80-400μs wasted per token just on barriers.

#### Required Fix: Custom Decode Attention Kernel

**Option A: GEMV-style attention (recommended for decode)**
```wgsl
// Treat attention as batched GEMV: Q[1,256] × K[256,kvLen] → scores[1,kvLen]
// Single thread per head, vectorized dot product
@compute @workgroup_size(256, 1, 1)  // One thread per head dim element
fn attention_decode_gemv(...) {
    // Parallel reduction across headDim
    // No barriers needed - each thread handles one element
}
```

**Option B: FlashAttention-2 style**
- Online softmax with block-wise processing
- Requires complete rewrite of attention kernel
- Higher complexity but better for long sequences

**Option C: Subgroup-based reduction**
- Use `subgroupAdd` for dot products
- Eliminates explicit barriers
- Requires subgroup support (already detected)

#### Investigation Summary (Dec 20, 2025)

**Attempted Approaches:**
1. ❌ GEMV-style with tiling - Still had barriers for KV position iteration
2. ❌ Flash Attention online softmax - Increased barriers to 1000+ due to block-wise processing
3. ❌ Simplified optimized variant - Still had more barriers than current kernel

**Root Cause:** Any approach that processes KV cache in blocks/tiles introduces barriers for synchronization. The current `attention_small_f16kv` kernel processes KV positions in blocks of 32, requiring barriers between blocks.

**Correct Solution:** Subgroup-based reduction (Option C)

**Algorithm for seqLen=1 decode:**
```wgsl
// For each head (workgroup size = headDim, one workgroup per head)
@compute @workgroup_size(256, 1, 1)  // headDim threads
fn attention_decode_subgroup(head_idx: u32) {
  let tid = global_id.x % 256;  // Thread within head
  let q_val = Q[head_idx][tid];  // Each thread loads one Q element

  // Compute attention scores (no barriers - subgroup ops only)
  for (let k = 0; k < kvLen; k++) {
    let k_val = K_cache[k][head_idx][tid];
    let dot = q_val * k_val;
    let score = subgroupAdd(dot);  // Parallel reduction, no barrier
    scores[k] = score / sqrt(headDim);  // Only thread 0 writes
  }

  // Softmax (subgroup parallel scan)
  let max_score = subgroupMax(scores[tid]);
  let exp_val = exp(scores[tid] - max_score);
  let sum_exp = subgroupAdd(exp_val);
  scores[tid] = exp_val / sum_exp;

  // Weighted sum over V (no barriers)
  let output_val = 0.0;
  for (let k = tid; k < kvLen; k += 256) {
    let v_val = V_cache[k][head_idx][tid];
    output_val += scores[k] * v_val;
  }
  output[head_idx][tid] = subgroupAdd(output_val);
}
```

**Key Properties:**
- Zero workgroup barriers (only subgroup operations)
- 100% thread utilization (all headDim=256 threads active)
- Naturally parallelizes across heads
- Requires subgroup support (already available on M3)

#### Files Modified

| File | Change | Status |
|------|--------|--------|
| `gpu/kernels/wgsl/attention_decode_subgroup.wgsl` | ✅ Created | Subgroup-optimized decode kernel (4 barriers vs 80) |
| `gpu/kernels/attention.ts` | ✅ Modified | Added subgroup kernel selection, caps detection |
| `gpu/kernels/utils.ts` | ✅ Modified | Registered decode_subgroup variant |
| `inference/pipeline/attention.ts` | ⬜ TODO | Will auto-route via runAttention |

#### Expected Speedup

| Current (attention_small_f16kv) | With Subgroup Decode Kernel |
|--------------------------------|----------------------------|
| 80 barriers/token | **4 barriers/token** (20x reduction) |
| 31/32 threads idle (97% idle) | 256/256 threads active (100%) |
| 72ms attention/token (measured) | ~10-20ms attention/token (est.) |
| 86% of GPU time | ~20-40% of GPU time |

**Barrier Analysis:**
- Current kernel: 80 barriers (8 head tiles × 2 barriers × 5 KV blocks)
- New kernel: 4 barriers (softmax coordination: after scores, after max, after exp, before output)
- Reduction: 20x fewer barriers

**Expected overall impact:** 3.6-7.2x speedup in attention → **~2-3x end-to-end speedup** (4 tok/s → 8-12 tok/s)

**Status:** ✅ Implemented (Dec 20, 2025) - ready for testing

**Priority:** P0 (highest impact - attention is 86% of GPU time)
- Medium complexity (subgroup operations + shared memory coordination)
- Clean separation: only affects decode path (seqLen=1)
- Auto-enabled when hasSubgroups=true on device
- Synergy with Section 1.20 (GPU-only decode loop) for maximum benefit

**Next Steps:**
1. Test kernel with Gemma 1B (headDim=256)
2. Benchmark actual speedup vs current kernel
3. Validate numerical correctness (attention scores, output)
4. Profile barrier overhead reduction

### 1.23 Performance Regression Investigation (Dec 2025) ⚠️

**Status:** Active investigation - 50% performance regression identified

#### Benchmark Results (2025-12-19)

| Model | Layout | Actual | Expected | Gap |
|-------|--------|--------|----------|-----|
| gemma-1b-q4-col | column_wise | **4.0 tok/s** | 8.0 tok/s | -50% ❌ |
| gemma-1b-q4-flat | flat | **4.0 tok/s** | 7.0 tok/s | -43% ❌ |
| gemma-1b-q4-row | row-wise (fused) | 3.0 tok/s | 3.0 tok/s | ✅ Match |

**Current decode latency:** ~265ms/token
**Target decode latency:** ~125ms/token (for 8 tok/s)

#### Key Findings

**What's Working ✅:**
- Kernel configuration correct - manifests have proper hints
- Fused Q4K performs as documented - row-wise achieves 3 tok/s
- Prefill is excellent - 79-82 tok/s
- GPU features enabled - F16 ✓, Subgroups ✓, Timestamp Query ✓
- Results are consistent - multiple runs stable at 4 tok/s

**Issue Identified ❌:**

The dequant path is 50% slower than documented. This affects BOTH column-wise and flat equally:
- ❌ NOT a layout issue (both layouts affected)
- ❌ NOT a configuration issue (hints correct)
- ❌ NOT a GPU capability issue (features enabled)
- ✅ IS a kernel performance issue (dequant or GEMV)

#### Root Cause Analysis

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| GPU submit overhead | 6.3 submits/token (should be 1-2) | ⚠️ Investigate |
| Dequant kernel regression | Both layouts affected equally | ⚠️ Investigate |
| GEMV kernel regression | Column-wise not faster than flat | ⚠️ Investigate |
| Memory bandwidth saturation | Unknown | ⬜ Profile |

#### Investigation Tasks (P0)

| Task | Status | Notes |
|------|--------|-------|
| Profile dequant kernel (GB/s throughput) | ⬜ TODO | Use `gpu/profiler.ts` |
| Profile GEMV kernel (FLOPS achieved) | ⬜ TODO | Measure actual vs theoretical |
| Reduce GPU submit count | ⬜ TODO | Target: 1-2 submits/token |
| Check git history for regressions | ⬜ TODO | Compare with previous commits |
| Compare with WebLLM on same hardware | ⬜ TODO | Baseline comparison |
| Allow GEMV for column-major weights | ✅ Done | Removed transposeB gating in `gpu/kernels/matmul.ts` |

#### Files Created During Investigation

- `PERFORMANCE_INVESTIGATION_2025_12_19.md` - Deep technical analysis
- `KERNEL_FIXES_2025_12_19.md` - Configuration fixes applied
- `BENCHMARK_RESULTS_2025_12_19.md` - Detailed benchmark analysis
- `tools/diagnose-kernels.ts` - Diagnostic tool

#### Immediate Optimization Targets

1. **Optimize dequant kernel** - Better subgroup usage
2. **Optimize GEMV kernel** - Memory coalescing
3. **Implement command batching** - Fewer GPU submits (6.3 → 1-2)
4. **Add kernel+epilogue fusion** - matmul + SiLU

### 1.24 Infrastructure & Quality (P1)

**Goal:** Ensure long-term project health and prevent regressions.

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| **Add unit tests** | P1 | ⬜ TODO | Currently mostly Playwright E2E. Need granular unit tests for `gpu/*.ts` logic. |
| **CI pipeline** | P1 | ⬜ TODO | Automated testing on GitHub Actions (needs GPU runner or mock). |
| **Performance regression tracking** | P1 | ⬜ TODO | Track tok/s over time to catch regressions like 1.23. |
| **Artifact size budget** | P2 | ⬜ TODO | Monitor bundle size impact of new kernels. |

---

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Decode tok/s (Gemma 3 1B, M3) | >= 40 | **~4** | ⬜ (10x gap) ⚠️ |
| Per-token latency | <= 25ms | **~265ms** | ⬜ (10x gap) ⚠️ |
| Time to first token | <= 800ms | **~650ms** | ✅ Achieved |
| VRAM usage vs WebLLM | <= 110% | ~980MB (Q4K) | ✅ |
| Readback bytes/token | <= 4KB | 4 bytes | ✅ (fused argmax) |

**Dec 2025 Benchmark (M3 MacBook):**

| Variant | Decode tok/s | TTFT | VRAM |
|---------|--------------|------|------|
| Q4K column_wise | 8.0 | 650ms | 979MB |
| Q4K flat | 7.0 | 700ms | 965MB |
| Q4K row_wise | 3.0 | 1600ms | 992MB |
| F16 | 9.4 | 540ms | 1.9GB |

**WebLLM comparison (WeInfer paper):**
- WebLLM on Qwen2-1.5B: 24.18 ms/token (~41 tok/s)
- DOPPLER on Gemma 3 1B: ~125 ms/token (~8 tok/s)
- Gap: ~5x (was 10x before optimizations, was 6x before column_wise)

**Measurement commands:**
```bash
# Quick performance check
npm run doppler -- bench inference --prompt xs --headed

# Check if fused Q4K is actually used
npm run doppler -- bench inference --prompt xs 2>&1 | grep -i "falling back"

# View detailed results
cat doppler/tests/results/*.json | jq '.metrics'
```

---

## Key Files

| File | Purpose |
|------|---------|
| `gpu/command-recorder.ts` | Command buffer batching |
| `gpu/buffer-pool.ts` | Buffer reuse |
| `gpu/profiler.ts` | GPU timestamp profiling |
| `gpu/kernels/*.wgsl` | WGSL shader sources |
| `gpu/kernels/utils.ts` | Kernel configs (incl. `gemv_subgroup_multicol`) |
| `gpu/kernels/matmul.ts` | Matmul kernel selection, layout handling, multicol dispatch |
| `gpu/kernels/matmul_q4_fused.wgsl` | Fused Q4K dequant+matmul (GEMV + multicol + batched) |
| `gpu/kernels/matmul_gemv_subgroup.wgsl` | F16 GEMV (4-col + 32-col multicol variants) |
| `gpu/kernels/sample.ts` | GPU argmax kernel (`recordArgmax` for batching) |
| `gpu/kernel-selector.ts` | Kernel dispatch |
| `gpu/kernel-tuner.ts` | Workgroup auto-tuning |
| `inference/pipeline.ts` | Forward pass, fused decode path (lines 686-737) |
| `inference/pipeline/logits.ts` | `recordLogitsGPU` for batched logits computation |
| `inference/pipeline/ffn.ts` | FFN with gate+up fusion support |
| `loader/doppler-loader.ts` | Model loading, `useFusedQ4K` flag |
| `tools/rdrr-writer.ts` | Weight transpose, gate+up fusion |
| `tools/convert-cli.ts` | Converter with `--q4k-layout` flag |
| `storage/rdrr-format.ts` | Manifest types, layout metadata |

---

## Dependencies

None - this is the foundational phase.

---

## Next Phase

[Phase 2: MoE Efficiency](PHASE_2_MOE.md) - Requires buffer reuse and async pipeline from Phase 1.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
