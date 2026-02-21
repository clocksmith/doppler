# BF16 Runtime Math Spike

This note captures practical BF16 strategies for Doppler given current WebGPU/WGSL constraints.

## Context

- Doppler supports BF16 as a manifest/storage dtype.
- Runtime math paths are F16/F32 kernel families.
- WebGPU feature detection currently uses `shader-f16`; no BF16 compute feature is wired.

## Repro Script

Run:

```bash
node tools/bf16-math-spike.js
```

The script reports:

1. Dot-product accuracy across precision policies
2. Error drift under different BF16 rounding cadence
3. F16 vs BF16 overflow points under multiplicative growth
4. RMSNorm stability across activation scales
5. Softmax stability under logit growth (naive vs max-subtracted)

## Current Spike Results

Using the deterministic defaults in `tools/bf16-math-spike.js`:

- Dot study (`trials=300`, `width=2048`)
  - `bf16Strict` mean abs err vs f32: `~0.1002`
  - `bf16InputF32Acc` mean abs err vs f32: `~0.0056`
  - `f16Strict` mean abs err vs f32: `~0.6702`
  - `f16InputF32Acc` mean abs err vs f32: `~0.0017`
- Overflow study (`start=1`, `growth=1.13`)
  - F16 overflow step: `90`
  - BF16 overflow step: `725`
- RMSNorm study (`trials=120`, `width=4096`, `scale=128`, `eps=1e-6`)
  - `f16Strict` mean abs err vs f32: `~0.7979`
  - `f16InputF32Acc` mean abs err vs f32: `~1.46e-4`
  - `bf16Strict` mean abs err vs f32: `~0.3394`
  - `bf16InputF32Acc` mean abs err vs f32: `~1.12e-3`
- Softmax study (`trials=250`, `width=256`, `scale=16`)
  - `f16StrictNaive` non-finite vector rate: `1.0`, mean exp overflow/trial: `~62.5`
  - `f16StrictStable` non-finite vector rate: `0.0`, mean prob-sum abs err: `~7.56e-4`
  - `f16InputF32AccStable` non-finite vector rate: `0.0`, mean prob-sum abs err: `~1.45e-8`

Interpretation:

- Strict 16-bit accumulation is much worse than 16-bit input + F32 accumulation.
- BF16's exponent range matters a lot for overflow-heavy regimes.
- RMSNorm and softmax both confirm that widening accumulation/normalization paths buys disproportionate stability.
- Mixed precision policies are the most practical quality/perf lever.

## Mathematically Sound Strategy Options

### 1) Exact BF16 arithmetic emulation (bit-exact)

- Represent values as BF16 payloads (`u16`, packed in `u32`).
- Implement unpack/operate/normalize/round (RNE) with IEEE handling.
- Correctness: strongest (can be bit-exact).
- Cost: highest instruction count and control-flow pressure.

### 2) BF16 inputs + F32 accumulate + BF16/F16 output at boundaries

- Decode BF16 once per load tile.
- Perform multiply-accumulate in F32.
- Re-quantize only when writing persistent state.
- Correctness: strong for ML workloads.
- Cost: moderate; usually much better than strict emulation.

### 3) Deferred rounding windows for state updates

- Keep intermediate activation/state in F32 for N ops.
- Re-quantize to BF16/F16 at explicit boundaries.
- Correctness: often much better than per-op requantization.
- Cost: more transient F32 bandwidth/state.

Evidence from this spike:

- RMSNorm at high activation scale shows a large gap between strict F16 and F16-input + F32 accumulation.
- Stable softmax with F32 accumulation stays finite and nearly normalized where strict naive F16 collapses.

### 4) Range-aware selective widening (recommended baseline)

- Keep vulnerable ops/paths in F32:
  - residual adds
  - norm
  - softmax/logits critical regions
- Keep bounded ops in F16 where stable.
- Correctness: high for known overflow paths.
- Cost: targeted increase, usually acceptable.

## Doppler-Oriented Implementation Path

1. Add a precision policy layer (experimental runtime preset only):
   - `strict16`
   - `input16_f32acc`
   - `selective_f32_windows`
2. Instrument per-op overflow/NaN counters in debug mode.
3. Tune window boundaries for Gemma 3:
   - residual/norm in F32
   - matmul outputs where safe by kernel path
4. Compare throughput and error against existing `f16a` and `f32a` kernel paths.

## Practical Takeaway

The best near-term approach is not "BF16 everywhere in-kernel" but:

- keep 16-bit storage where it helps bandwidth,
- use F32 where overflow/accumulation error dominates,
- and minimize round-trip requantization frequency.
