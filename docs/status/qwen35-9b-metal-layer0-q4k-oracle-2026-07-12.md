# Qwen 3.5 9B M3 layer-0 Q4K projection oracle

The independent scalar Q4_K dequantization and matrix multiplication agrees
with Doppler's optimized Metal output for `layer.0.ffn.gate`. The first
material F16-to-Q4 difference is therefore real quantization loss in the
stored layer-0 gate weights, not a packed-layout, scale/min decode, or
optimized-kernel error at this projection.

This result narrows the mixed-Q4 correctness failure but does not yet explain
why the full model immediately selects `<|im_end|>`.

## Fixed comparison

The oracle uses prompt-token row 30 from the same deterministic 31-token
prompt used by the accepted F16 control and failing mixed-Q4 candidate. It
freezes all 4,096 values at `layer.0.ffn.in`, then computes the 12,288-output
gate projection four ways:

1. Doppler's Q4 Metal kernel with the mixed-Q4 activation;
2. an independent scalar Q4_K decoder and matrix multiplication with that
   identical activation;
3. Doppler's F16 Metal projection with the F16 activation; and
4. a scalar F16 matrix multiplication using both the F16 activation and the
   frozen Q4 activation.

The Q4 tensor is `model.language_model.layers.0.mlp.gate_proj.weight`, shape
`[12288, 4096]`, dtype `Q4_K_M`, row layout. Its exact 28,311,552 packed bytes
have SHA-256
`f3f9c62827addc9c92d9af9ffa2b1e9e3252141ac5606b050a606ef8aa61bd10`.
The reference decoded 196,608 blocks, including 1,572,864 scales and minima.

## Result

| Comparison | Maximum absolute difference | RMSE | Cosine similarity |
| --- | ---: | ---: | ---: |
| Q4 Metal vs scalar Q4 reference | `0.0000008344650268554688` | `0.00000006354751818538317` | `0.9999999999998528` |
| F16 Metal vs scalar F16 reference | `0.0000025033950805664062` | `0.00000013297147344603175` | `0.9999999999994142` |
| Scalar Q4 vs scalar F16, same frozen input | `0.05161425657570362` | `0.007551034730242163` | `0.9981637906100974` |
| Q4 Metal vs F16 Metal | `0.05160520039498806` | `0.007551061583841115` | `0.9981637792457753` |

The Q4 and F16 input activations differ by at most
`0.000024825334548950195`. Recomputing the F16 projection with the frozen Q4
input changes its output by at most `0.00003793835639953613`. That input drift
is too small to account for the observed `0.0516` Q4-to-F16 gate difference.

The Q4 Metal output and scalar reference differ by less than one millionth at
their worst element. The scalar Q4 and scalar F16 calculations reproduce the
much larger GPU-to-GPU difference. This disproves the test hypothesis that
Doppler's optimized layer-0 gate kernel materially disagrees with a trusted
decode of the stored Q4 bytes.

## What is closed and what remains open

Closed for this exact layer-0 gate projection:

- Q4_K scale/min decoding mismatch;
- row-layout or packed-byte interpretation mismatch;
- optimized Q4 projection-kernel corruption; and
- F16 projection-kernel corruption.

Still open:

- whether quantization error accumulates across later Q4 projections until it
  changes the first-token ranking;
- whether one or more other quantized tensors have a separate conversion or
  execution defect; and
- which weights must remain F16 for the mixed artifact to recover coherent
  base-token parity.

The mixed-Q4 artifact remains rejected for base-model correctness. Adapter
activation and performance tuning remain gated. The V12 AMD/ROCm training
matrix was not changed or rerun.

## Reproduction and evidence

The machine-readable oracle is
`reports/qwen-3-5-9b-metal/2026-07-12/layer0-ffn-gate-oracle.json`. Its SHA-256
is `c2110ea9edd3b5c0e78e9b5423f4cec5a93777f18d2d1b75cbebf9ef78a08f6d`.
It records the host and Metal adapter, Doppler commit, source revision,
manifest and artifact identities, runtime configuration, prompt and token
IDs, exact command, comparisons, and hashes for every binary sidecar.

```sh
node tools/q4k-projection-oracle.js \
  --q4-model-dir ../../rdrr/qwen-3-5-9b-q4k-ehaf16 \
  --f16-model-dir ../../rdrr/qwen-3-5-9b-f16-af32 \
  --q4-capture reports/qwen-3-5-9b-metal/2026-07-12/q4-layer0-ffn-gate-full.json \
  --f16-capture reports/qwen-3-5-9b-metal/2026-07-12/f16-layer0-ffn-gate-full.json \
  --tensor model.language_model.layers.0.mlp.gate_proj.weight \
  --input-op layer.0.ffn.in \
  --output-op layer.0.ffn.gate \
  --row 30 \
  --artifact-dir reports/qwen-3-5-9b-metal/2026-07-12/layer0-ffn-gate-oracle \
  --out reports/qwen-3-5-9b-metal/2026-07-12/layer0-ffn-gate-oracle.json
```

The implementation is covered by a synthetic Q4_K encoder/decoder cross-check,
scalar projection checks, and the complete Doppler unit/integration suite.
