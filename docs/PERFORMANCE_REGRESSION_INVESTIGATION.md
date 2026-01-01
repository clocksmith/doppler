# Performance Regression Investigation (Tier 2 P0)

## Overview

**Current Performance:** ~4 tok/s
**Target Performance:** 40+ tok/s
**Gap:** 10x slower than target

This document investigates the performance bottlenecks causing the 10x performance gap and provides actionable fixes.

## Executive Summary

The 10x performance gap is caused by:
1. **Kernel launch overhead** - Too many small kernel dispatches per token
2. **Memory bandwidth saturation** - Redundant buffer reads/writes between kernels
3. **Suboptimal decode kernels** - Not leveraging M=1 optimizations
4. **Missing kernel fusion** - Separate gate/up projections waste bandwidth

## Profiling Methodology

### GPU Timing

Use the benchmark harness to measure individual kernel timings:

```typescript
import { benchmarkDecodePass, printBenchmarkReport } from './gpu/kernel-benchmark.js';

const report = await benchmarkDecodePass({
  modelConfig: {
    hiddenSize: 1152,
    intermediateSize: 6912,
    numHeads: 4,
    numKVHeads: 1,
    headDim: 256,
    vocabSize: 262144,
    numLayers: 26,
  }
});

printBenchmarkReport(report);
```

### Expected Breakdown (Gemma 3 1B)

| Operation | Expected (ms) | Current (ms) | Notes |
|-----------|---------------|--------------|-------|
| RMSNorm (x2) | 0.02 | 0.2 | 10x overhead |
| QKV Matmul | 0.1 | 0.5 | Not using GEMV |
| Attention | 0.2 | 2.0 | Full softmax recompute |
| Out Proj | 0.05 | 0.3 | Not using GEMV |
| FFN Gate+Up | 0.15 | 1.5 | Separate kernels |
| FFN Down | 0.1 | 0.5 | Not using GEMV |
| LM Head | 0.5 | 3.0 | Large vocab (262K) |
| **Total/layer** | ~1.1 | ~8.0 | **7x gap** |
| **26 layers** | ~28 | ~208 | |
| **tok/s** | ~35 | ~5 | **7x gap** |

## Identified Bottlenecks

### 1. Kernel Launch Overhead (30% of gap)

**Problem:** Each decode token requires ~200+ kernel dispatches.

**Evidence:**
- RMSNorm: 2 per layer = 52 dispatches
- Matmul: 4 per layer = 104 dispatches
- Attention: 1 per layer = 26 dispatches
- Activation: 1 per layer = 26 dispatches
- Total: ~208 dispatches per token

**Fix:** Use CommandRecorder to batch all layer operations into a single submit.

```typescript
// Before: Multiple submits
for (const layer of layers) {
  await runRMSNorm(...);  // submit
  await runMatmul(...);   // submit
  await runAttention(...); // submit
}

// After: Single submit
const recorder = new CommandRecorder(device);
for (const layer of layers) {
  await recordRMSNorm(recorder, ...);
  await recordMatmul(recorder, ...);
  await recordAttention(recorder, ...);
}
recorder.submit();
```

**Expected Improvement:** 1.5-2x

### 2. Memory Bandwidth Waste (25% of gap)

**Problem:** Separate gate/up projections read input twice.

**Evidence:**
```
FFN gate: Read input (1152 * 4 bytes) + Read W_gate + Write output
FFN up:   Read input (1152 * 4 bytes) + Read W_up + Write output
Total: 2x input reads, 2x output writes
```

**Fix:** Use fused FFN kernel (`ffn_fused.wgsl`).

```typescript
// Before
const gate = await runMatmul(input, W_gate, 1, 6912, 1152);
const up = await runMatmul(input, W_up, 1, 6912, 1152);
const activated = await runSiLU(up, { gate });

// After
const activated = await runFusedFFN(input, W_gate, W_up, 1152, 6912, {
  activation: 'silu'
});
```

**Expected Improvement:** 1.3-1.5x for FFN pass

### 3. Non-optimized Decode Matmul (20% of gap)

**Problem:** Using batched matmul kernels for M=1 decode.

**Evidence:**
- Generic matmul: 16x16 tiles, many idle threads for M=1
- GEMV kernel: 256 threads, optimized for single-row

**Fix:** Ensure matmul.ts selects GEMV variant for M=1:

```typescript
// In selectMatmulVariantAndFlags
if (M === 1 && effectiveBDtype === 'f16' && aDtype === 'f32') {
  if (capabilities.hasSubgroups) {
    variant = 'gemv_subgroup_multicol';  // For large N
  } else {
    variant = 'gemv';
  }
}
```

**Expected Improvement:** 2-3x for projections

### 4. Attention Decode Overhead (15% of gap)

**Problem:** Using prefill-style attention for single-token decode.

**Evidence:**
- Prefill kernel: Tiles over seqLen, unnecessary for seqLen=1
- Full softmax stored: Wastes shared memory

**Fix:** Use optimized decode attention kernel (`attention_decode_optimized.wgsl`):

- Online softmax (no full score storage)
- Vectorized KV cache reads
- Subgroup reductions

**Expected Improvement:** 3-4x for attention

### 5. LM Head Bottleneck (10% of gap)

**Problem:** 262K vocab matmul dominates decode time.

**Evidence:**
- LM head: 1152 x 262144 = 302M MACs per token
- Memory read: 262144 * 1152 * 2 bytes (F16) = 604MB

**Fix:**
1. Use multi-column GEMV (`gemv_subgroup_multicol`)
2. Consider top-k logit computation (skip full softmax)
3. Weight-tied embeddings can share memory

**Expected Improvement:** 1.5-2x

## Implementation Checklist

### Phase 1: Quick Wins (Day 1)

- [x] Ensure GEMV kernel selection for M=1 matmuls
- [x] Enable CommandRecorder for batched execution
- [x] Verify subgroup operations are being used

### Phase 2: Kernel Fusion (Day 2-3)

- [x] Implement fused FFN kernel
- [x] Integrate into pipeline
- [x] Benchmark before/after

### Phase 3: Attention Optimization (Day 4-5)

- [x] Implement optimized decode attention
- [x] Add online softmax
- [x] Benchmark against baseline

### Phase 4: LM Head (Day 6)

- [ ] Profile LM head performance
- [ ] Implement weight-tied optimization
- [ ] Consider partial logit computation

## Profiling Commands

```bash
# Run kernel benchmarks
doppler test kernels --perf

# Full inference benchmark
doppler test inference --perf

# Enable kernel debug logging
DOPPLER_DEBUG_KERNELS=true doppler test inference
```

## Success Metrics

| Metric | Current | Target | Achieved |
|--------|---------|--------|----------|
| tok/s | 4 | 40+ | TBD |
| Per-layer latency | 8ms | 1ms | TBD |
| Total decode latency | 200ms | 25ms | TBD |

## Appendix: Kernel Timings Reference

### Gemma 3 1B (M1 Pro, WebGPU)

Expected optimal timings:
- RMSNorm (1x1152): 0.01ms
- MatMul GEMV (1x1152x1152): 0.05ms
- MatMul GEMV (1x6912x1152): 0.1ms
- MatMul GEMV (1x262144x1152): 0.5ms
- Attention decode (4 heads, 256 dim, 512 KV): 0.1ms
- SiLU (6912 elements): 0.01ms

Total per layer: ~0.4ms
26 layers + LM head: ~11ms
tok/s: ~90

With overhead and non-optimal paths: ~40 tok/s target is achievable.
