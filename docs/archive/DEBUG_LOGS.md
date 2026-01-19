# Debug Logs & Archives
## Debug Sessions

**Last Updated**: 2025-12-17 23:13 UTC
**Status**: UNSOLVED - See [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](postmortems/2025-12-17-positive-bias-hidden-states.md)

## Quick Start

1. **Invoke the doppler-debug skill** when investigating inference issues
2. **Run quick benchmarks** to reproduce:

```bash
# Reproduce the garbage output bug
doppler bench inference --config debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN|blue|Kaw|Generated"

# Look for: ALL POSITIVE values at last position
# FINAL_HIDDEN[pos=6]: [183.x, 42.x, 201.x, ...] - ALL POSITIVE (bug!)
```

## Current Issue (UNSOLVED)

**Prompt**: "The color of the sky is"
**Expected**: "blue"
**Actual**: "Kaw" (garbage token 44821)

| Observation | Value | Status |
|-------------|-------|--------|
| FINAL_HIDDEN[pos=0] | [-97, -21, -76, ...] Mixed | Correct |
| FINAL_HIDDEN[pos=6] | [183, 42, 201, ...] ALL POSITIVE | **BUG** |
| Token "Kaw" logit | 28.35 (MAX) | Wrong |
| Token "blue" logit | 4.81 | Should be higher |

**Root cause**: Hidden states at last token position are all positive, causing tokens with positive embeddings to dominate.

## Priority Investigation

1. **Q4_K dequantization verification** - Does GPU kernel actually produce negative values?
2. **Layer-by-layer tracking at pos=N-1** - When does positive bias start?
3. **Reference comparison** - Run llama.cpp with same weights and compare

See postmortem for full hypothesis ranking and next steps.

## Log Levels (verbosity)

Control general log verbosity via runtime config:

| Config | Level | Shows |
|--------|-------|-------|
| `runtime.shared.debug.logLevel.defaultLogLevel=info` | info | Phase starts/ends, totals |
| `...=verbose` | verbose | + Per-shard source (RAM/OPFS/network), per-layer timing |
| `...=silent` | silent | Errors only |

## Trace (categories)

Trace is separate from log level. Use runtime config for tensor/kernel details:

- `runtime.shared.debug.trace.enabled=true`
- `runtime.shared.debug.trace.categories=["kernels","attn"]`
- `runtime.shared.debug.trace.categories=["all","-buffers"]`

**Defaults by preset:**
- `bench`: log=info, trace off
- `debug`: log=verbose, trace on (all categories)

**Config-only usage:** `runtime.shared.debug.trace` is the source of truth.

## Config-Driven Probes (Preferred for Readbacks)

Use probes to read specific token/dimension values without adding ad-hoc logs:

```bash
# Post-softcap logits probe (Gemma 2 parity)
npm run debug -- --config '{
  "runtime": {
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["logits"] },
        "probes": [
          { "id": "topk", "stage": "logits_final", "tokens": [-1], "dims": [476, 3868] }
        ]
      }
    }
  }
}'
```

Probes run on CPU or GPU buffers; they are skipped when CommandRecorder batching is active.

### Common Grep Patterns

```bash
# Debug with verbose loader output
doppler debug 2>&1 | grep -E "Loader.*Shard|Loader.*Layer" | head -50

# Layer-by-layer debug output
doppler debug 2>&1 | grep -E "Layer[0-9]" | head -50

# Full debug with logits and generated text
doppler debug 2>&1 | grep -E "Layer|logits|top-5|Generated" | head -50

# Position-specific hidden state debug
doppler debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN" | head -20

# Trace-level output (tensor details)
doppler debug --config debug 2>&1 | head -200
```

**If logs don't appear:** Check your grep pattern includes the tag (e.g., `Loader` for loader output).

## Debug Flag

**IMPORTANT:** Debug GPU readbacks are gated behind `runtime.shared.benchmark.run.debug=true` or the `debug` preset to avoid performance impact.

- Without flag: Benchmarks run at full speed (no GPU sync points)
- With flag: Verbose layer-by-layer output but much slower

```bash
# Fast benchmark (no debug output)
doppler bench inference --config bench --headed

# Slow benchmark with debug GPU readbacks
doppler bench inference --config debug
```

```typescript
// Programmatic debug
await pipeline.generate(prompt, { debug: true, maxTokens: 10 });
```

## Selective Layer Debugging (Faster)

Use `debugLayers` to debug only specific layers while keeping batching enabled for other layers:

```typescript
// Full debug (slow): syncs at EVERY layer
await pipeline.generate(prompt, { debug: true });

// Selective debug (faster): syncs only at checkpoint layers
await pipeline.generate(prompt, {
  debug: true,
  debugLayers: [0, 12, 25],  // First, middle, last layers
});
```

This dramatically speeds up debug runs by:
1. Keeping CommandRecorder enabled for non-checkpoint layers
2. Only flushing GPU commands and reading back hidden states at specified layers
3. Recreating the recorder after each checkpoint to continue batching

For Gemma 3 1B (26 layers), typical checkpoint choices:
- `[0]` - Only first layer (embedding issues)
- `[25]` - Only final layer (pre-logits state)
- `[0, 12, 25]` - First, middle, last (balanced)
- `[0, 1, 2, ..., 25]` - All layers (same as `debug: true` alone)

## OPFS Cache Persistence (Faster Reruns)

The benchmark runs inside a persistent Playwright profile directory. This preserves browser storage between runs, including the OPFS model cache.

- Default inference benchmark profile: `doppler/.benchmark-cache/`
- Override with `--profile-dir <path>` (relative to `doppler/` or absolute)

```bash
# Warm run (reuse existing OPFS cache)
doppler bench inference --config bench --profile-dir .benchmark-cache

# Cold run (fresh profile dir)
doppler bench inference --config bench --profile-dir .benchmark-cache-cold
```

## CommandRecorder Gotcha

**CRITICAL**: When using CommandRecorder (batched mode), debug readbacks show zeros!

Always check `!recorder` before attempting debug buffer reads:
```typescript
if (layerIdx === 0 && !recorder) {
  // Safe to debug readback
} else if (recorder) {
  console.log('(skipping - batched mode)');
}
```

## Key Files

- `inference/pipeline.js` - decode loop
- `inference/pipeline/layer.js` - layer processing with debug readbacks
- `inference/pipeline/logits.js` - final norm + lm_head
- `gpu/kernels/matmul_q4_fused.wgsl` - Q4_K dequantization kernel
- `inference/pipeline/probes.js` - config-driven probe readbacks
- `config/schema/debug.schema.js` - trace/probe schema

## Performance Context

Target: 40+ tok/s decode on Gemma 3 1B. See `feature-log/doppler/inference.jsonl` for task tracking.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Performance Regression Investigations

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

Use the kernel benchmark harness to measure per-kernel timings:

```bash
# Kernel microbenchmarks
doppler test kernels --perf

# Full inference benchmark (for tok/s + latency)
doppler test inference --perf
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

**Fix:** Ensure matmul.js selects GEMV variant for M=1:

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

# Enable kernel trace logging
doppler test inference --config debug
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


## Test Results

Index of DOPPLER validation sessions across different hardware and browsers.

## Quick Status

| Platform | Status | Last Tested |
|----------|--------|-------------|
| Apple M3 (macOS) | Working | Dec 2025 |
| AMD Strix Halo (Linux) | Blocked (headless) | Dec 2025 |
| NVIDIA | Untested | - |

---

This file is a human-readable log. Store machine-readable benchmark outputs as JSON using
`style/BENCHMARK_STYLE_GUIDE.md` so results can be compared automatically.

See also:
- `style/BENCHMARK_STYLE_GUIDE.md` for benchmark methodology and JSON result schema
- `TESTING.md` for WGSL kernel testing specification
- `../tests/kernels/README.md` for kernel test coverage
- `../tests/kernels/BENCHMARKS.md` for kernel microbenchmark baselines

## Result Artifacts (Recommended)

| Artifact | Purpose | Suggested Path |
|----------|---------|----------------|
| Pipeline benchmark JSON | TTFT, tok/s, submits, readback, memory | `tests/results/` |
| Kernel correctness JSON/HTML | per-kernel correctness | `tests/kernels/test-results/` |
| Kernel benchmark JSON/HTML | per-kernel timings | `tests/kernels/test-results/` |
| Baseline registry | Expected tok/s ranges | `tests/baselines.json` |

If a run does not have a JSON artifact yet, record the session here and file it as follow-up work.

## Test Sessions

### Session 2025-12-14: AMD Strix Halo + Gemma 3 1B

**Tester**: Linux AMD machine
**GPU**: AMD Strix Halo integrated GPU (Radeon 8050S/8060S Graphics)
**Browser**: Google Chrome 142.0.7444.162
**OS**: Linux 6.17.0-7-generic
**Model**: Gemma 3 1B IT (Q4_K_M quantization)

#### Status: IN PROGRESS

**Steps completed**:
1. ✓ Located Gemma 3 1B model in HuggingFace cache
2. ✓ Converted to RDRR format (Q4_K_M quantization) - 965MB, 15 shards
3. ✓ Model served locally for browser load
4. ☒ **BLOCKED**: Headless browser cannot access WebGPU (no GPU in headless environment)

**Test Limitation**: The Linux environment runs headless without X server or GPU access. WebGPU requires either:
- A headed browser with GPU drivers (X11/Wayland + working GPU)
- OR Manual testing in a desktop environment

**Model is ready** - just needs a desktop browser to test.

**Model path**:
- Source: `/home/clocksmith/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- RDRR output: `./models/gemma-3-1b-q4/`

**Expected model size**: ~1.2GB (340 tensors, 26 layers, 1152 hidden size)

---

### Session 2025-12-14: Mac M3 + GPT-OSS 20B (parallel session)

**Tester**: MacBook with M3
**GPU**: Apple M3 (unified memory)
**Model**: GPT-OSS 20B MoE (Q4_K_M, 32 experts, topK=4)
**Status**: PARTIAL - Router fixed, expert loading in progress

#### Bug Fix 1: MoE Gather Kernel (FIXED)
- Root cause: WebGPU `layout: 'auto'` only includes bindings used by each entry point
- `count_and_map` used 4/6 bindings, `gather_tokens` used 6/6
- Bind group creation with mismatched layout caused silent failure
- Fix: Created explicit bind group layout with all 6 bindings
- See: `postmortems/2025-12-22-moe-explicit-layout.md`

#### Bug Fix 2: Router Weight Quantization (FIXED)
- Root cause: Router weights quantized to Q4_K_M despite HuggingFace config `modules_to_not_convert`
- Symptom: Router logits extreme (56 vs -39), softmax collapses to [1.0, 0.0, 0.0, 0.0]
- Fix: Updated `shouldQuantize()` in quantizer.js to check:
  1. Hard-coded `router` and `gate.weight` patterns
  2. HuggingFace `modules_to_not_convert` config from quantization_config
- Reconverted model: Router weights now BF16 (184KB vs 52KB Q4_K_M)
- **Result**: Router now produces distributed weights!
  ```
  [DEBUG MoE L0] Router logits (first 8 experts): -0.14, 0.59, 0.11, 0.88, -0.98, -1.73, 2.58, -0.54
  [DEBUG MoE L0] Expert weights: [0.5896, 0.1668, 0.1359, 0.1078, ...]
  ```
  vs before: `[1.0, 0.0, 0.0, 0.0]`

#### Current Status: Expert Loading
- Router works correctly (distributed weights)
- Expert tensor loading attempted but using wrong naming convention
- GPT-OSS uses packed MXFP4 experts (`model.layers.X.mlp.experts.gate_up_proj_blocks`)
- Loader fallback exists but may need debugging

**Files modified**:
- `gpu/kernels/moe.js` - Added explicit bind group layout for MoE
- `gpu/kernels/moe_gather.wgsl` - Cleaned up, added layout note
- `src/converter/quantizer.js` - Added router check in `shouldQuantize()`
- `src/converter/node-converter.js` - Pass `modules_to_not_convert` to shouldQuantize

---

## Hardware Configurations Tested

| Date | GPU | VRAM | Browser | OS | Model | Status | Notes |
|------|-----|------|---------|----|----|-------|-------|
| 2025-12 | Apple M3 | Unified | Safari/Chrome | macOS | Gemma 3 1B | ✓ WORKING | Reference implementation |
| 2025-12-14 | AMD Strix Halo | Integrated | Chrome 142 | Linux | Gemma 3 1B | ⏳ TESTING | In progress |
| 2025-12-14 | Apple M3 | Unified | Chrome | macOS | GPT-OSS 20B | ⚠️ PARTIAL | MoE pipeline works, output quality poor |

## Test Protocol

### Standard Result Capture (Recommended)

For each performance session, record:

- Model: `modelId`, quantization, shard count, tensor count
- Environment: OS, browser version, GPU adapter info, WebGPU feature flags
- Workloads: prompt names and token counts
- Metrics: TTFT, prefill tok/s, decode tok/s, peak VRAM estimate, GPU submit counts
- Output quality: `quality.ok` plus reasons/warnings

Preferred output:

- A JSON file per run matching `style/BENCHMARK_STYLE_GUIDE.md`
- A short narrative summary in this document for context and troubleshooting.
Baseline ranges live in `tests/baselines.json` and are enforced in CI when enabled.

To avoid instruction drift, prefer linking to the canonical runner docs:

- Kernel tests and microbenchmarks: `../tests/kernels/README.md` and `../tests/kernels/BENCHMARKS.md`
- End-to-end inference tests: `tests/harness.html` (set `runtime.shared.harness.mode` via `runtimeConfig`)

## Known Issues by Platform

### AMD GPUs
- **Driver requirements**: Mesa 23.0+ (Linux) or Adrenalin 23.0+ (Windows)
- **WebGPU status**: Generally good support in recent drivers
- **Strix Halo**: New integrated RDNA architecture, untested

### Apple Silicon
- **Unified memory advantage**: No PCIe overhead, can load larger models
- **Safari vs Chrome**: Both support WebGPU, Safari may have better integration
- **F16 support**: Excellent on M-series chips

### NVIDIA
- **Status**: Untested in DOPPLER
- **Expected**: Should work well with recent drivers
- **Driver**: 525+ required for WebGPU

## Debugging Common Issues

### Model loads but produces garbage tokens
**Symptom**: Output like `<unused16>`, random Unicode, or non-English text for English prompts

**Causes**:
1. Quantization format mismatch (Q4_K encoding issue)
2. BF16 conversion error
3. Gemma 3 norm offset not applied
4. GPU dequantization kernel bug

**Debug**:
- Check logs for "Prefill logits" top-5 distribution
- Look for negative hidden state values (should be present)
- Compare against known-working Mac M3 output

### WebGPU not available
**Symptom**: `navigator.gpu` is undefined

**Solutions**:
- Update browser (Chrome 113+, Edge 113+)
- Enable in Firefox: `about:config` → `dom.webgpu.enabled`
- Check GPU drivers are up to date

### Out of memory errors
**Symptom**: Buffer allocation fails, model won't load

**Solutions**:
- Try smaller model (Gemma 3 1B needs ~1.2GB)
- Close other GPU-intensive apps
- Check browser console for specific buffer size limits

## Contributing Results

After testing:
1. Update this file with your results
2. Update HARDWARE_COMPATIBILITY.md matrix
3. Commit and push changes
4. Share any issues or findings

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.
