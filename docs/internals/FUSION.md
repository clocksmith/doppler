# Kernel Fusion Internals

Technical deep-dive on kernel fusion opportunities, command batching, and optimization matrix.

---

## Currently Implemented Fusions

| Fusion | Passes | Speedup | Files |
|--------|--------|---------|-------|
| Gate+Up FFN | 3→2 | 1.2-1.3x | `ffn.js`, `src/converter/writer.js` |
| FlashAttention (tiled + online softmax) | N→1 | 2x | `attention.wgsl` |
| Logits+Argmax+Sampling | 3→1 | 1.3-1.5x | `logits.js`, `sample.wgsl` |
| Dequant Q4K → F16 GEMV | 2 | 2.3x | `dequant_subgroup.wgsl`, `matmul_gemv_subgroup.wgsl` |
| Multi-column Q4K GEMV | - | 8x fewer wg | `matmul_q4_fused.wgsl:main_multicol` |
| Subgroup GEMV | - | 1.5x | `matmul_gemv_subgroup.wgsl` |
| Command buffer batching | 260→1 submits | 50-100ms | `command-recorder.js` |

---

## Residual+RMSNorm Fusion

**Status:** Implemented - Actual impact ~1.5% (minimal because RMSNorm <1% of GPU time)

**Before:**
```typescript
// 2 separate kernel passes
let postAttn = await doRMSNorm(attnOutput, normWeight, eps, {...});
postAttn = await doResidualAdd(postAttn, inputBuffer, {...});
```

**After:**
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
// rmsnorm.wgsl - rmsnorm_inplace_residual with shared memory cache
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

**Key Finding:** Minimal impact because RMSNorm is **<1% of GPU time**. The real bottleneck is attention (86% of GPU time at 67-73ms/token).

---

## Missing Fusion Opportunities

### P0 - High Impact

| Fusion | Current | Proposed | Est. Speedup | Complexity |
|--------|---------|----------|--------------|------------|
| **Quantized Matmul+RMSNorm** | 2-3 passes | 1 pass | 1.2-1.5x | Medium |

### P1 - Medium Impact

| Fusion | Current | Proposed | Est. Speedup | Complexity |
|--------|---------|----------|--------------|------------|
| **Matmul+RMSNorm Epilogue** | 2 passes | 1 pass | 1.1-1.3x | Medium |
| **Attention+Residual** | 2 passes | 1 pass | 1.1-1.2x | Medium |

### P2 - Lower Priority

| Fusion | Current | Proposed | Est. Speedup | Complexity |
|--------|---------|----------|--------------|------------|
| **Matmul+SiLU Epilogue** | 2 passes | ~1.5 passes | 1.1-1.2x | Medium |
| **Parallel FFN Gate+Up+Down** | Sequential | Concurrent | 1.1-1.3x | Medium |
| **Parallel Q/K/V Projection** | Sequential | Concurrent | 1.1-1.2x | Medium |

---

## F16 Activation Pipeline (Implemented)

**Current:** F16 activations are supported end-to-end when `shader-f16` is available.
**Precision rules:**
- F16 buffers for hidden states and matmul outputs
- F32 internal accumulation for RMSNorm/softmax
- F16 sampling/logits when configured (readback still handled safely)

**Status:** Implemented for Gemma 2/3 paths; see `docs/postmortems/2026-01-05-gemma2-f16-end-to-end.md`.
**Remaining risk:** Numerical underflow in deep layers on some devices; validate with probes when introducing new kernels.

---

## Parallel Kernel Execution

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
- `gpu/command-recorder.js` - Add `recordParallelGroup()` method
- Group kernels share same command encoder pass
- WebGPU can overlap execution of independent compute dispatches

---

## GPU-Only Decode Loop (N Tokens Without CPU Roundtrip)

**Goal:** Generate 5-100 tokens on GPU without reading back to CPU between tokens.

### Current Flow (1 readback per token)

```
Token 1: GPU forward → submit → readback → JS sampling → GPU forward
Token 2: GPU forward → submit → readback → JS sampling → GPU forward
...
Token N: GPU forward → submit → readback → JS sampling → done

Total: N submits, N readbacks, N JS→GPU roundtrips
At 265ms/tok with 6.3 submits/tok = ~42ms per submit overhead
```

### Proposed Flow (1 readback per N tokens)

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

### Expected Speedup

| Current | With GPU Loop (N=10) | With GPU Loop (N=100) |
|---------|---------------------|----------------------|
| 6.3 submits/tok | 0.63 submits/tok | 0.063 submits/tok |
| 265ms/tok | ~50-80ms/tok | ~30-50ms/tok |
| 4 tok/s | 12-20 tok/s | 20-33 tok/s |

---

## Complete Optimization Matrix

### Implemented (Baseline: ~4 tok/s)

| # | Optimization | Speedup | Cumulative |
|---|--------------|---------|------------|
| 1 | Column-wise Q4K layout | 2.7x | 2.7x |
| 2 | FlashAttention fusion | 2x | 5.4x |
| 3 | Subgroup GEMV | 1.5x | 8.1x |
| 4 | GPU sampling (no readback) | 1.3-1.5x | 10-12x |
| 5 | Gate+Up FFN fusion | 1.2-1.3x | 12-16x |
| 6 | Command buffer batching | - | - |
| 7 | F16 KV cache | - | - |
| 8 | BF16→F16 weights | 1.2-1.5x | - |

### Theoretical Maximum

**Step 1: Fix regression (P0) - recover 2x**
```
Current: 4 tok/s (265ms/tok)
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

---

## Key Files

| File | Purpose |
|------|---------|
| `gpu/command-recorder.js` | Command buffer batching |
| `gpu/profiler.js` | GPU timestamp profiling |
| `inference/pipeline.js` | Forward pass, fused decode path |
| `inference/pipeline/logits.js` | `recordLogitsGPU` for batched logits |
| `inference/pipeline/ffn.js` | FFN with gate+up fusion |
