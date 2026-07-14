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
| Manifest SHA-256 | `ac1eff7f371b7d676a5b9a7ddcf94e2d01d3e1043f499dede2fc41249ef55956` | `2ff1f1eec0345bd379614b8b0bafd66957bbec1c22072132661a6e35300ad398` |
| Shard-set hash | `sha256:27c1ba6d379e43eff7eae12b6ccc83081fe5248d1739e7877e82428d52daf228` | `sha256:7d4053a21a45d06690a1e4ac3377cd4d2d46a7239ce94f739bb2e5830f16900d` |
| Weight-pack hash | `sha256:dfeea5dc362804bd197d1f96effea74c4cd803eb93b8304c633057f4f172f237` | `sha256:d337596cd6ae6cb360b60bca56d95b2e24f5253113953455b6565ffcd055ee46` |
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

Raw receipts and logits are under
`reports/qwen-3-5-9b-metal/2026-07-14/`.
