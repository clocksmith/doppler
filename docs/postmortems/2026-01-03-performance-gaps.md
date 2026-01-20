# Performance Gap Analysis

**Date:** 2026-01-03
**Status:** Open (performance gap remains; F16 pipeline now end-to-end for Gemma 2)
**Context:** DOPPLER vs WebLLM comparison on Gemma 2 2B Q4_K_M

**Update (2026-01-11):** F16 activations now run end-to-end on Gemma 2/3 when `shader-f16` is available (gather, matmul/GEMV, attention, RoPE, sampling). This closes Issue 1 below; see `2026-01-05-gemma2-f16-end-to-end.md`.

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

### Status: RESOLVED (F16 activations end-to-end with shader-f16)

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

### Resolution (2026-01-11)

F16 activations now run end-to-end on supported devices:

1. **Embedding output** - `gather_f16*` variants emit F16 hidden states
2. **Matmul/GEMV** - F16 output variants selected for decode and LM head
3. **Attention/RoPE/Sampling** - F16 paths selected when activations/KV are F16

Example:

```bash
doppler --config <ref>
# <ref>: extends "debug", cli.command="debug", model="gemma-2-2b-it-wf16"
# runtime.inference.prompt="Explain why the sky is blue."
# runtime.inference.batching.maxTokens=8
# runtime.inference.chatTemplate.enabled=true
# runtime.shared.debug.trace.enabled=true, categories=["kernels"]
```

Expected trace: `gather_f16*`, `attention_small_f16`, `sample_f16`, and GEMV F16 variants.

---

## Issue 2: Fused Q4K 2.3x Warning

### Evidence

```typescript
// matmul.js:509
trace.kernels(`Q4K FUSED: ... (WARNING: 2.3x slower than dequant)`);

// doppler-loader.js:169
useFusedQ4K = false;  // Default is dequant-first
```

### Status: CONFIGURABLE (kernel-path driven)

Auto-selection now prefers fused Q4K when subgroups + F16 activations are available
(`gemma2-q4k-fused-f16a`). Dequant-first paths remain available (`gemma2-q4k-dequant-f16a/f32a`)
and can be forced via kernel path overrides for perf comparisons.

### Trade-off Matrix

| Mode | Speed | VRAM | Use Case |
|------|-------|------|----------|
| Dequant-first | Fast (baseline) | High | Small models (≤4B) |
| Fused | Device-dependent | Low | Large models (9B+) |

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
| F16 activations | 2x bandwidth | High | ✓ RESOLVED |
| Batching bug | 37% speedup | Medium | ✓ FIXED |
| Fused Q4K | N/A | None | Configurable |
| Streaming fallback | N/A | None | Not triggered |

## Batching Bug Fix (2026-01-03)

**Root cause:** Uniform buffer cache eviction destroyed buffers still referenced by pending command buffers.

**Fix applied:**
1. Added `pendingDestruction` queue to `UniformBufferCache`
2. Evicted buffers are deferred, not destroyed immediately
3. `flushPendingDestruction()` called after GPU work completes in `submitAndWait()` and `pipeline.js`
4. Fixed `isDecodeMode` flag in batched decode path

**Result:** Batched decode now works correctly. Output is coherent (no more garbage/repetition).

## Remaining Performance Gap

After batching fix, decode gap remains; profiling shows matmul and attention dominate decode time.

| Factor | Contribution | Status |
|--------|--------------|--------|
| F16 activations | Bandwidth | ✓ RESOLVED |
| Matmul throughput | High | Needs kernel tuning |
| Attention throughput | Medium | Needs kernel tuning |

## Q4K Strategy Fix (2026-01-06)

**Problem:** gemma2.json preset effectively forced the fused Q4K path (legacy `q4kStrategy: "fused_q4k"`), which forces the slow on-the-fly dequant path for ALL Q4K matmuls (2.3x slower than dequant-first).

**Fix:** Use `kernelPath: "gemma2-q4k-dequant-f16a"` for Gemma 2 presets.

**Result:**
| Metric | Before (fused_q4k) | After (dequant_f16a) | Improvement |
|--------|-------------------|---------------------|-------------|
| Overall | ~1.3 tok/s | ~9 tok/s | 6.9x |
| Pure decode | N/A | ~12-13 tok/s | - |

**Updated gap analysis:**
- DOPPLER decode: ~12-13 tok/s
- WebLLM decode: ~20.3 tok/s
- Gap: **1.6x** (down from perceived 30x due to wrong Q4K strategy)

**Update (2026-01-11):** Kernel paths now control Q4K selection directly. Use
`runtime.inference.kernelPath` to force `gemma2-q4k-fused-f16a` or
`gemma2-q4k-dequant-f16a` depending on device performance.

## F16 Activations: Resolved

Original dtype propagation issues are resolved. See
`2026-01-05-gemma2-f16-end-to-end.md` for the verified F16 path
and debug commands.
