# Debug Session: Positive Bias at Last Token Position

**Last Updated**: 2025-12-17 23:13 UTC
**Status**: UNSOLVED - See [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](postmortems/POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md)

## Quick Start

1. **Invoke the doppler-debug skill** when investigating inference issues
2. **Run quick benchmarks** to reproduce:

```bash
# Reproduce the garbage output bug
doppler bench inference --prompt xs --debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN|blue|Kaw|Generated"

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

Control general log verbosity with CLI flags:

| CLI Flag | Level | Shows |
|----------|-------|-------|
| (default) | info | Phase starts/ends, totals |
| `--verbose` | verbose | + Per-shard source (RAM/OPFS/network), per-layer timing |
| `--quiet` | silent | Errors only |

## Trace (categories)

Trace is separate from log level. Use it for tensor/kernel details:

- `--trace` (all categories)
- `--trace kernels,attn`
- `--trace all,-buffers`

**Defaults by mode:**
- `bench`: log=info, trace off
- `debug`: log=verbose, trace on (all categories)

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
doppler debug --trace 2>&1 | head -200
```

**If logs don't appear:** Check your grep pattern includes the tag (e.g., `Loader` for loader output).

## Debug Flag

**IMPORTANT:** Debug GPU readbacks are gated behind `--debug` or `debug: true` to avoid performance impact.

- Without flag: Benchmarks run at full speed (no GPU sync points)
- With flag: Verbose layer-by-layer output but much slower

```bash
# Fast benchmark (no debug output)
doppler bench inference --prompt xs --headed

# Slow benchmark with debug GPU readbacks
doppler bench inference --prompt xs --debug
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
doppler bench inference --prompt xs --profile-dir .benchmark-cache

# Cold run (fresh profile dir)
doppler bench inference --prompt xs --profile-dir .benchmark-cache-cold
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

- `inference/pipeline.ts` - decode loop
- `inference/pipeline/layer.ts` - layer processing with debug readbacks
- `inference/pipeline/logits.ts` - LAST_TOKEN_HIDDEN debug
- `gpu/kernels/matmul_q4_fused.wgsl` - Q4_K dequantization kernel
- `inference/pipeline/sampling.ts` - specific token logit tracking

## Performance Context

Target: 40+ tok/s decode on Gemma 3 1B. See `feature-log/doppler/inference.jsonl` for task tracking.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
