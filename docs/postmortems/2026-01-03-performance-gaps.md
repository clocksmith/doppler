# Performance Gap Analysis

**Date:** 2026-01-03
**Status:** Open (F16 partially implemented, batching workaround applied)
**Context:** DOPPLER vs WebLLM comparison on Gemma 2 2B Q4_K_M

## Current State

| Metric | WebLLM | DOPPLER | Gap |
|--------|--------|---------|-----|
| Prefill | 52.6 tok/s | 26.0 tok/s | 2x |
| Decode | 20.3 tok/s | 7-8 tok/s | 2.5x |

## Issue 1: F32 Activations (Bandwidth)

### Evidence

```typescript
// matmul.js:79
outputDtype = 'f32',  // DEFAULT IS F32
```

### Impact

- Weights: F16 (or Q4K→F16 dequant)
- Activations: F32 (4 bytes/element)
- Bandwidth per token: 26 layers × 2304 hidden × 4 bytes = **238 KB/token**
- With F16: 26 × 2304 × 2 = **119 KB/token** (2x reduction)

### Status: PARTIALLY IMPLEMENTED (fundamental architecture issue discovered)

### Completed

1. ✓ Added `activationDtype: 'f16' | 'f32'` to `RuntimeConfigSchema.inference.compute`
2. ✓ Created `f16-activations.json` runtime preset
3. ✓ Wired `activationDtype` through `LayerContext` and `AttentionConfig`
4. ✓ Updated matmul calls to pass `outputDtype: context.activationDtype`
5. ✓ Created `rmsnorm_f16.wgsl` - F16 input/output variant with F32 intermediate precision
6. ✓ Created `silu_f16.wgsl` - F16 SiLU/SwiGLU/GeGLU variants (including `geglu_rowsplit_f16`)
7. ✓ Updated `rmsnorm.js` kernel selector to use F16 variants when `activationDtype='f16'`
8. ✓ Updated `silu.js` kernel selector to use F16 variants when `activationDtype='f16'`
9. ✓ Updated `gelu.js` to use `geglu_rowsplit_f16` when `activationDtype='f16'`
10. ✓ Updated `decode-buffers.js` to allocate F16 buffers when configured
11. ✓ Enabled `outputDtype: context.activationDtype` in all FFN matmul calls in `layer.js`
12. ✓ Added `activationDtype` to `doRMSNorm` and all 6 call sites in `layer.js`
13. ✓ Updated RMSNorm to support F16 with residual (via has_residual uniform)
14. ✓ Updated attention.js o_proj matmul to use `outputDtype: config.activationDtype`
15. ✓ Added `canUseF16()` to RMSNorm to check actual buffer dtypes and fall back gracefully

### F16 Shader Files
- `src/gpu/kernels/rmsnorm_f16.wgsl` - RMSNorm with F16 I/O, F32 reduction precision
- `src/gpu/kernels/silu_f16.wgsl` - SiLU, SiLU-gate, SwiGLU rowsplit, GeGLU rowsplit with F16 I/O

### Fundamental Architecture Issue (discovered 2026-01-03)

The pipeline cannot use F16 activations because the embedding stage is F32-only:

1. **Embedding (gather.js)** - Always outputs F32, no F16 output variant
2. **Matmul mixed-precision** - `f16w_f32a` kernel (F16 weights + F32 activations) outputs F32, not F16
3. **Cascade effect** - Layer 0 receives F32, so all RMSNorms and kernels fall back to F32

```
Embedding → F32 → Layer0 input_norm → F32 → Q/K/V (f16w_f32a) → F32 → ...
                                              ^-- outputs F32 even with outputDtype='f16'
```

The `canUseF16()` check in RMSNorm correctly detects F32 buffers and falls back to F32 shaders,
preventing garbage output but also preventing F16 benefit.

### Fixes Required for True F16 Activations

1. **Add F16 output to gather.wgsl** - New variant that converts F32 embeddings to F16 output
2. **Add F32→F16 matmul variant** - Or add post-matmul cast kernel
3. **Update scale.wgsl** - F16 variant for embedding scaling

### Current Behavior with `--config f16-activations`

All operations fall back to F32 due to dtype propagation. Output is correct (no garbage) but
no bandwidth reduction is achieved.

### Benchmark

```bash
# Currently falls back to F32 throughout:
npm run debug -- -m gemma-2-2b-it --config f16-activations

# Future: After gather.wgsl F16 support:
npm run bench -- --filter matmul --config f16-activations
```

---

## Issue 2: Fused Q4K 2.3x Warning

### Evidence

```typescript
// matmul.js:509
trace.kernels(`Q4K FUSED: ... (WARNING: 2.3x slower than dequant)`);

// doppler-loader.js:169
useFusedQ4K = false;  // Default is dequant-first
```

### Status: ✓ ALREADY OPTIMAL

The warning only fires when `useFusedQ4K=true` (opt-in for VRAM-constrained scenarios). Default behavior is dequant-first which is 2.3x faster.

### Verification

Device log shows:
```
[Loader] Initialized (f16, subgroups, unified)
```

No "Q4K FUSED" trace messages in output - confirms dequant path is active.

### Trade-off Matrix

| Mode | Speed | VRAM | Use Case |
|------|-------|------|----------|
| Dequant-first | Fast (default) | High | Small models (≤4B) |
| Fused | 2.3x slower | Low | Large models (9B+) |

---

## Issue 3: Streaming Fallback

### Evidence

```typescript
// attention.js:122-123
log.warn('Attention', `No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). Falling back to streaming. Expect slow prefill.`);
tier = 'streaming';
```

### Status: ✓ NOT TRIGGERED

Device has subgroups support:
```
[GPU] apple metal-3, f16/subgroups, 4.0GB
```

Gemma 2 2B config: `headDim=256, numHeads=8`
- Subgroup kernel supports up to headDim=256 ✓
- Falls through to `tier = 'subgroup'` (line 110)

### Tier Selection Logic

```
if (canSubgroup) → 'subgroup'  // ← Gemma 2 uses this
else if (canLarge) → 'tiled_large'
else if (canSmall) → 'tiled_small'
else → 'streaming'  // Slow fallback
```

### When Streaming Triggers

- headDim > 256 without subgroups
- Device lacks sufficient shared memory for tiled kernels
- Would see warning: "Expect slow prefill"

---

## Priority Matrix

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| F16 activations | 2x bandwidth | High | P2 (requires gather.wgsl F16) |
| Batching bug | 37% speedup | Medium | P1 ✓ FIXED |
| Fused Q4K | N/A (already fast) | None | Done |
| Streaming fallback | N/A (not triggered) | None | Done |

## Batching Bug Fix (2026-01-03)

**Root cause:** Uniform buffer cache eviction destroyed buffers still referenced by pending command buffers.

**Fix applied:**
1. Added `pendingDestruction` queue to `UniformBufferCache`
2. Evicted buffers are deferred, not destroyed immediately
3. `flushPendingDestruction()` called after GPU work completes in `submitAndWait()` and `pipeline.js`
4. Fixed `isDecodeMode` flag in batched decode path

**Result:** Batched decode now works correctly. Output is coherent (no more garbage/repetition).

## Remaining Performance Gap

After batching fix, current decode speed: ~5.5 tok/s vs WebLLM 20.3 tok/s (2.7x gap)

| Factor | Contribution | Status |
|--------|--------------|--------|
| F32 activations | ~30% | Blocked (needs Tensor abstraction) |
| Batching | N/A | ✓ FIXED |
| Unknown | ~40% | Needs GPU profiling |

## Q4K Strategy Fix (2026-01-06)

**Problem:** gemma2.json preset effectively forced the fused Q4K path (legacy `q4kStrategy: "fused_q4k"`), which forces the slow on-the-fly dequant path for ALL Q4K matmuls (2.3x slower than dequant-first).

**Fix:** Use `kernelPath: "q4k-dequant-f16"` (legacy `q4kStrategy: "dequant_f16"`) for Gemma 2 presets.

**Result:**
| Metric | Before (fused_q4k) | After (dequant_f16) | Improvement |
|--------|-------------------|---------------------|-------------|
| Overall | ~1.3 tok/s | ~9 tok/s | 6.9x |
| Pure decode | N/A | ~12-13 tok/s | - |

**Updated gap analysis:**
- DOPPLER decode: ~12-13 tok/s
- WebLLM decode: ~20.3 tok/s
- Gap: **1.6x** (down from perceived 30x due to wrong Q4K strategy)

## F16 Activations: Architecture Issue (2026-01-03)

**Problem:** The `--config f16-activations` preset produces garbage output because:
1. Embedding (gather.js) outputs F32
2. Downstream kernels try to use F16 variants based on config
3. Dtype mismatches cause silent data corruption

**Root cause:** Dtype is tracked implicitly (WeakMap + runtime checks) rather than structurally.

**Attempted fixes:**
- Added `canUseF16()` checks to kernels → still broken
- Fixed decode-buffers.js to allocate F32 sizes → still broken
- Multiple dtype propagation paths exist, each needs fixing

**Solution: Tensor Abstraction Layer**

Created `src/gpu/tensor.js`:
```typescript
interface Tensor {
  buffer: GPUBuffer;
  dtype: 'f16' | 'f32';
  shape: readonly number[];
}
```

**Migration plan:**
1. Migrate gather.js → returns Tensor with explicit dtype
2. Migrate rmsnorm.js, silu.js, matmul.js → Tensor input/output
3. Update layer.js to pass Tensors through pipeline
4. Remove WeakMap dtype tracking

**Benefits:**
- Dtype structurally required (can't forget)
- Mismatches caught at operation boundaries
- Enables future dtypes (BF16, INT8)

**Current state:** F16 preset disabled (uses F32 fallback).

**Next steps:**
1. Complete Tensor abstraction migration
2. Add F16 output to gather.wgsl
3. Profile with GPU timestamps for remaining bottlenecks
