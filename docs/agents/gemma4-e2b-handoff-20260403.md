# Gemma 4 E2B Inference Handoff — 2026-04-03

## Goal

Get `gemma-4-e2b-it-q4k-ehf16-af32` producing coherent text in Doppler. HF reference (BF16, transformers 5.6.0.dev0) predicts "The" for prompt "The color of the sky is" (temp=0, topK=1). Doppler currently predicts "csvStream" — garbled output.

## Artifact

Local artifact at `/tmp/gemma4-e2b-verify-20260403/` with manifest refreshed from `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json`. 98 shards, Q4K weights, F16 embeddings. Manifest schema is `doppler.execution/v1`.

## What was committed (9d935ef8)

**attention.valueNorm field** — Gemma 4 applies a unit-scale RMSNorm to V before attention (HF `v_norm` with `with_scale=False`). Doppler had the helper `applyAttentionValueNorm()` in `projections.js` but never called it. The fix adds `valueNorm: boolean` to the manifest schema, merge logic, param categories, validation, and both conversion configs. `run.js` now calls the helper when `config.valueNorm === true`. `record.js` previously called it unconditionally for all models; now gated behind the config flag.

**Post-attention sandwich norm separation** — In `layer.js`, the `else if (sandwichNorm.useSandwichNorm ...)` branch was changed from fused `doRMSNorm(..., { residual })` to separate `doRMSNorm` then `doResidualAdd`. This is mathematically equivalent — the fused kernel already does post-norm residual add (`result = rmsnorm(x) + residual`), not pre-norm (`rmsnorm(x + residual)`). The separation is harmless but not the fix it was intended to be.

## What was verified correct against HF

- **L0 projections/RoPE**: q_norm, k_norm, q_rope, k_rope all match within bf16 noise.
- **L0 FFN output**: Doppler `[5.87, 31.70, -4.01, ...]` vs HF `[5.91, 32.5, -4.13, ...]`. Close, Q4K precision.
- **Embedding scaling**: main `sqrt(1536)=39.19`, per-layer `sqrt(256)=16`. Both correct.
- **Tokenization**: 15 tokens, IDs match HF exactly (`[2, 105, 2364, ...]`).
- **RoPE config**: local theta=10000, rotaryDim=256; global theta=1M, rotaryDim=128, freqBaseDim=512. Matches HF proportional RoPE.
- **Layer_scalar**: F16 weights load correctly. L0=0.0178, L1=0.223, etc. Match HF.
- **Per-layer-input architecture**: gate→GeLU→multiply→project→norm→residual matches HF forward.
- **RMSNorm fused residual**: Kernel does `rmsnorm(x) + residual` (post-norm add). Correct for Gemma 4.
- **Execution path**: Pipeline degrades (ops not in PIPELINE_COMPATIBLE_OPS). Standard `processLayerGPU` path is used, not the plan path. `layer_out` probes fire from `ffn/sandwich.js`, not the plan path.
- **Attention geometry**: headDim correctly toggles between 256 (sliding) and 512 (full) per layer type.
- **Decode strategy**: `replay_prefill` due to mixed headDim + shared KV. No KV cache allocated.

## What is still broken

The model outputs garbled text. This predates the v_norm commit — confirmed by testing at `a4fb853d`.

## Unverified hypotheses (ranked by likelihood)

1. **Per-layer-input Q4K noise amplification** — gate/projection weights are Q4K. The per-layer contribution is large (e.g., -20 at L0 dim 1), and the layer_scalar (0.018) amplifies relative error. Reconverting with F16 per-layer-input weights would test this.
2. **Full-attention layer numerical correctness** — headDim=512 attention has not been boundary-probed against HF. The streaming attention kernel at 512-dim heads may have issues.
3. **Per-layer embedding gather offset** — The shared embedding table [262144, 8960] is sliced per-layer via `hiddenOffset`. The gather kernel's offset handling has not been numerically verified.
4. **`layer_out` probe placement** — The probe fires from `ffn/sandwich.js` BEFORE `applyPerLayerInputBlock`. Probed values are pre-per-layer, pre-scalar. No probe exists for the actual layer output.

## Files touched

`src/config/` (schema, merge, params, validation, conversion configs), `src/inference/pipelines/text/attention/` (run, record, projections, types), `src/inference/pipelines/text/` (config, layer), `tests/` (4 test files).
