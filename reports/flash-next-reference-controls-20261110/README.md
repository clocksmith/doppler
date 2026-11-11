# Flash-Next inventory gate: reference controls

The 2,187-token `inventory` prompt still **fails** the unchanged
`doppler.boundary-tolerance/quantized-v1` gate. These two controls ask whether
that failure could come from noise in the CPU reference itself. It does not:
both independent CPU references agree with the original within the gate, and
the GPU stays outside it against all of them.

| Comparison (last position, 248,320 logits) | Max absolute error | Violations | Greedy token |
| --- | ---: | ---: | --- |
| GPU vs original F32 CPU (16 threads) | 0.467271 | **30,664** | 271 / 271 |
| F32 CPU, 8 threads vs 16 threads | 0.000011 | 0 | 271 / 271 |
| F32 CPU (16 threads) vs F64-widened CPU | 0.027117 | 0 | 271 / 271 |
| GPU vs F64-widened CPU | 0.458771 | **30,729** | 271 / 271 |

`marbles` (22 tokens) passes every comparison; its largest error is 0.000027.
Full rows, RMS errors and payload hashes are in [comparison.json](comparison.json).

## What this establishes

- The CPU reference is stable when its arithmetic order changes. Changing only
  the thread count moves the inventory logits by at most 1.1e-5.
- A higher-precision reference sides with the original F32 reference
  (0.027 maximum, inside the gate) and not with the GPU.
- The GPU's distance from the reference (0.46–0.47) is about 17 times larger
  than the F32-vs-F64 reference distance. The gate failure is real GPU
  deviation, so it is not grounds for a wider tolerance. The tolerance is
  unchanged.

This agrees with the earlier localization in
[flash-next-new-prompts-20261110](../flash-next-new-prompts-20261110/README.md):
GPU and CPU source agree to F32 rounding for every operator on identical inputs
in Layers 0–24. Accumulated upstream evaluation-order differences
(`4.59e-6` relative) then flip a near-tied rank-10 expert choice at
Layer 24, token 112 (expert 225 instead of 60, margin about `8e-6`). After
that flip, routing diverges chaotically: 9,754 inferred expert-set differences
through Layer 47.

## What it does not establish

- **Not an exact reference.** The F64 control widens the producer's explicit
  `.float()` casts (router logits, GDN gate, compressed-attention pooling and
  scores, LM head). Upstream source modules that cast to float32 internally
  are not widened.
- **No fix located.** The remaining owner is the GPU's accumulated upstream
  rounding in Layers 0–24, which is larger than the CPU F32 reference's own.
  Closing the gate requires reducing it, for example with more accurate
  accumulation in the upstream projections, so the Layer-24 margin is no
  longer crossed. No candidate change has been run yet.
- **No new GPU execution.** The GPU vector is the retained frozen F32-KV AMD
  observation from the new-prompts report.

## Reproduce

The producers differ from the pinned
`artifacts/qwen38-flash-next-20261013/prepare-learned-forward-reference.py`
(SHA-256 `f6bc0f15…`) only as shown in
[producer-threads8.diff](producer-threads8.diff) and
[producer-float64-widened.diff](producer-float64-widened.diff). Run each from
a private copy of that artifact directory, with the frozen
`inputs.json` (SHA-256 `3258ae23…`) and the converted artifact whose manifest
is SHA-256 `4902a0b7…`:

```sh
python3 -P producer/prepare-learned-forward-reference.py SOURCE_DIR OUT_T8 \
  --artifact ARTIFACT_DIR --inputs inputs.json
python3 -P producer-f64/prepare-learned-forward-reference.py SOURCE_DIR OUT_F64 \
  --artifact ARTIFACT_DIR --inputs inputs.json --dtype float64
```

On the AMD host, each run took 8 to 10 minutes on CPU at `nice 10`, beside
the active 27B provider. No service, GPU budget, runtime, artifact, tolerance
or catalog state changed.
