# Qwen 3.5 9B M3 base parity under Doppler 0.4.9

The exact mixed-Q4 artifact now produces coherent base inference under Doppler
0.4.9. With the same 31 input tokens, correctness profile, and greedy sampling,
Q4 and the accepted F16 control both generate `f32` with token IDs
`69, 18, 17, 248046`.

This reverses the earlier fixed-prompt Q4 rejection under Doppler 0.4.8. The
0.4.9 Q4 run selects `f` rather than `<|im_end|>`. A second Q4 run produces the
same selected token, generated tokens, output, and complete raw-logit hash.

## Fixed identities

| Field | Mixed Q4 | F16 control |
| --- | --- | --- |
| Source revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | same |
| Manifest, shard-set and weight-pack hashes | See [canonical status](qwen35-9b-metal-status.md#fixed-identities) | same |
| First token | `f` (`69`) | `f` (`69`) |
| Output | `f32` | `f32` |

The raw first-token logit vectors are finite but not identical. Their maximum
absolute difference is `2.761598229408264`, RMSE is
`0.5122713217379021`, and cosine similarity is
`0.9814944043975279`. The claim is coherent greedy-token parity for this
control, not exact numerical parity.

## Claim boundary

This receipt establishes base correctness for one fixed deterministic prompt
on the exact Q4 and F16 artifacts under Doppler 0.4.9. It does not promote the
Q4 artifact for general use, prove adapter activation, report a training gain,
or establish runtime performance. General base promotion still requires a
declared multi-prompt correctness population. The frozen AMD ROCm training
matrix was not changed or rerun.

Machine-readable evidence:
[qwen35-9b-metal-049-base-parity-2026-07-14.json](qwen35-9b-metal-049-base-parity-2026-07-14.json).

Raw receipts and logits were written to
`reports/qwen-3-5-9b-metal/2026-07-14/` and are not retained in the tree; their
SHA-256 values are in the JSON receipt. The canonical summary is
[qwen35-9b-metal-status.md](qwen35-9b-metal-status.md).
