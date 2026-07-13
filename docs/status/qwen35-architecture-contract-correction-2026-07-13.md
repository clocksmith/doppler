# Qwen 3.5 architecture-contract correction

The checked Qwen 3.5 conversion recipes and Doppler-native training graph used
the wrong post-attention normalization route. The generated v1 RDRR artifacts
therefore remain reproducible diagnostic artifacts, but they are no longer
valid base-model correctness controls. Adapter activation and performance work
remain blocked until corrected v2 artifacts pass a Transformers comparison.

This correction does not alter the frozen V12 AMD/ROCm experiment. V12 uses
the pinned Transformers Qwen implementation for training and evaluation.

## Reference contract

The authoritative reference is `transformers==5.13.1`
`Qwen3_5DecoderLayer` for `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The installed decoder source has
SHA-256
`cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba`.

The source implements this order:

```text
input norm -> token mixer -> attention residual
-> post-attention RMSNorm -> MLP -> MLP residual
```

It also rotates split halves inside the partial rotary prefix. The checkpoint's
`mrope_interleaved=true` controls multimodal frequency assignment; it does not
select adjacent-pair RoPE.

## Defects found

1. The native full- and linear-attention decoder modules normalized the
   attention output before adding the residual. Their scalar references copied
   the same error, so candidate/reference agreement could not detect it.
2. Qwen-specific training fixtures selected adjacent-pair RoPE. Their original
   two-dimension rotary slice made adjacent and split-half pairing identical,
   hiding the defect.
3. All four Qwen 3.5 conversion configs set
   `normalization.postAttentionNorm=true`. In Doppler this flag selects the
   Gemma sandwich-norm path, `hidden + RMSNorm(attention)`. Qwen requires the
   existing standard path, which uses the loaded `post_attention_layernorm`
   weight to compute `RMSNorm(hidden + attention)` before the FFN.

## Corrections

- Doppler commit `d20f1446` fixes forward/backward ordering in both decoder
  variants, updates the independent scalar references, and makes the Qwen
  fixtures exercise the production 0.25 partial-RoPE ratio with split-half
  pairing.
- Doppler commit `d61b76b0` pins the exact Transformers decoder source digest
  in the cross-backend fixture.
- Doppler commits `99fa6c54` and `df3f7fed` correct the four Qwen 3.5
  conversion configs and assign new `mv-exec-v2-norm-order` manifest variant
  identities. The weight-pack identities do not change.
- Gamma commits `452e51e` and `6d21d7b` replace the hand-copied parity decoder
  with the actual pinned Transformers `Qwen3_5DecoderLayer` and fail closed on
  source-digest or architecture-contract drift.

Scalar finite-difference tests for the corrected full and linear decoder
references pass. Conversion-contract, config single-source, agent-parity,
kernel-registry, kernel-digest, and JS/type export checks also pass. Clean GPU
requalification remains pending.

## Evidence impact

| Evidence | Disposition |
| --- | --- |
| V12 Gamma SFT and compiler evaluation | Unaffected; it executes the real Transformers model. |
| F16 projection-input-gradient, AdamW, accumulation-unit, and PEFT-format mechanics | Architecturally independent; retain their narrow claims. |
| Native full-attention, full-decoder, linear-decoder, hybrid-decoder, and integrated SFT receipts before this correction | Superseded; rerun against the corrected graph. |
| M3 v1 F16 output `` `f32` `` | Retained as a one-prompt smoke, not base-model correctness evidence. |
| M3 v1 mixed-Q4 immediate EOS | Retained as a reproducible failure under the v1 graph; rerun under v2 before causal attribution. |
| Layer-0 Q4 Metal/scalar oracle | Retains its kernel-local conclusion: the tested Q4 projection matched independent scalar dequantization for the supplied input. It does not validate the surrounding v1 model graph. |

## Required requalification

1. Rerun every affected native GPU oracle from a clean corrected revision.
2. Seal the rank-32 Gamma/Doppler microstep comparison against the pinned
   Transformers decoder.
3. Refresh F16 and mixed-Q4 manifests as distinct v2 artifacts without
   changing their weight packs.
4. Compare v2 F16 boundaries and logits against Transformers before treating
   it as a base control.
5. Re-evaluate mixed Q4 only after v2 F16 parity, then activate and compare a
   trained adapter only after coherent base inference.

Until those gates pass, Doppler-native Qwen training is mechanics work, not a
qualified production backend, and no v1 Qwen 3.5 artifact is eligible to prove
base inference correctness, adapter correctness, WGSL capability, or runtime
performance.

The machine-readable receipt is
[qwen35-architecture-contract-correction-2026-07-13.json](qwen35-architecture-contract-correction-2026-07-13.json).
