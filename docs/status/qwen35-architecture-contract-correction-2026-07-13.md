# Qwen 3.5 native-training architecture correction

The Doppler-native Qwen training graph and its tiny parity fixture used the
wrong decoder order and RoPE pairing. Those defects are corrected. A follow-on
hypothesis that the same normalization defect affected the checked RDRR
inference manifests was disproved by the manifest refresher and runtime audit;
the attempted inference-config change is retracted.

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

## Defects confirmed

1. The native full- and linear-attention decoder modules normalized the
   attention output before adding the residual. Their scalar references copied
   the same error, so candidate/reference agreement could not detect it.
2. Qwen-specific native-training fixtures selected adjacent-pair RoPE. Their
   original two-dimension rotary slice made adjacent and split-half pairing
   identical, hiding the defect.

## Inference hypothesis retracted

The initial follow-on audit incorrectly treated
`normalization.postAttentionNorm` as an algorithm selector. The manifest
refresher rejected the proposed `false` value because all 32
`post_attention_layernorm` tensors exist. Runtime inspection then established:

- `postAttentionNorm=true` records tensor presence;
- sandwich normalization is selected only when `preFeedforwardNorm` or
  `postFeedforwardNorm` is true;
- Qwen's existing `true/false/false` combination uses the standard path, which
  computes the attention residual and then applies the loaded post-attention
  RMSNorm before the MLP; and
- the v1 manifests resolve `ropeInterleaved` to false (split-half pairing),
  while `mropeInterleaved=true` remains a separate frequency-layout contract.

Therefore, the v1 RDRR manifests were not invalidated by the two native-training
defects. No v2 norm-order manifest is required, and the v1 weight-pack and
manifest-variant identities remain unchanged.

## Corrections

- Doppler commit `d20f1446` fixes forward/backward ordering in both native
  decoder variants, updates the independent scalar references, and makes the
  Qwen fixtures exercise the production 0.25 partial-RoPE ratio with split-half
  pairing.
- Doppler commit `d61b76b0` pins the exact Transformers decoder source digest
  in the cross-backend fixture.
- Gamma commits `452e51e` and `6d21d7b` replace the hand-copied parity decoder
  with the actual pinned Transformers `Qwen3_5DecoderLayer` and fail closed on
  source-digest or architecture-contract drift.
- Doppler commits `99fa6c54` and `df3f7fed` record the rejected inference
  hypothesis. The current correction restores `postAttentionNorm=true` and the
  original `mv-exec-v1` identities.

Scalar finite-difference tests for the corrected native full and linear decoder
references pass. Clean GPU requalification remains pending.

## Evidence impact

| Evidence | Disposition |
| --- | --- |
| V12 Gamma SFT and compiler evaluation | Unaffected; it executes the real Transformers model. |
| F16 projection-input-gradient, AdamW, accumulation-unit, and PEFT-format mechanics | Architecturally independent; retain their narrow claims. |
| Native full-attention, full-decoder, linear-decoder, hybrid-decoder, and integrated SFT receipts before the correction | Superseded; rerun against the corrected graph. |
| M3 v1 F16 output `` `f32` `` | Retains its original one-prompt F16-control scope. |
| M3 v1 mixed-Q4 immediate EOS and F16/Q4 boundary comparison | Retain their original base-inference diagnostic scope; the Q4 cause remains open beyond the layer-0 oracle. |
| Layer-0 Q4 Metal/scalar oracle | Retains its kernel-local conclusion: the tested Q4 projection matched independent scalar dequantization for the supplied input. |

## Required requalification

1. Rerun every affected native GPU oracle from a clean corrected revision.
2. Seal the rank-32 Gamma/Doppler microstep comparison against the pinned
   Transformers decoder.
3. Qualify matched-prefix accumulation, checkpoint/resume, and PEFT export.
4. Keep base inference, adapter inference, compiler capability, semantic kernel
   correctness, and runtime performance as separate gates.

Until those gates pass, Doppler-native Qwen training is mechanics work, not a
qualified production backend. This correction does not demote the separately
verified 0.8B/2B inference artifacts or broaden the M3 9B receipts beyond their
recorded prompts and artifact identities.

The machine-readable receipt is
[qwen35-architecture-contract-correction-2026-07-13.json](qwen35-architecture-contract-correction-2026-07-13.json).
