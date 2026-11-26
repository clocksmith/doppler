# Qwen 3.5 9B RDRR and M3 Metal status

This is the canonical status for the Qwen 3.5 9B RDRR artifacts and their
Apple M3 Metal correctness checks. The dated receipts listed in the
[chronology](#chronology) keep their original scope; where they disagree with
this file, this file is current.

Doppler versions here are versions of `doppler-gpu`, the package's former
name.

## Current verdict (2026-07-14)

Under Doppler 0.4.9 the exact mixed-Q4 artifact `qwen-3-5-9b-q4k-ehaf16`
produces coherent base inference on the fixed prompt. With the same 31 input
tokens, correctness profile and greedy sampling, Q4 and the F16 control
`qwen-3-5-9b-f16-af32` both generate `f32` with token IDs
`69, 18, 17, 248046`. A second Q4 run reproduces the selected token, the
generated tokens, the output and the complete raw-logit hash.

This reverses the fixed-prompt Q4 rejection recorded on 2026-07-11 and
2026-07-12 under the Doppler 0.4.8-era runtime, where Q4 selected `<|im_end|>`
first. The 0.4.9 release corrected split gated SiLU/SwiGLU, which had computed
`SiLU(gate) * SiLU(up)`; see the
[gated SiLU correction](gated-silu-runtime-correction-2026-07-13.md).

The raw first-token logit vectors of Q4 and F16 are finite but not identical
(maximum absolute difference `2.761598229408264`, RMSE `0.5122713217379021`,
cosine `0.9814944043975279`). The claim is coherent greedy-token parity on one
prompt, not exact numerical parity.

Claim boundary: one fixed deterministic prompt on the exact Q4 and F16
artifacts. This does not promote the Q4 artifact for general use, prove
adapter activation, report a training gain or establish runtime performance.
General base promotion still requires a declared multi-prompt correctness
population. The frozen AMD ROCm V12 training matrix was not changed or rerun.

## F16 control availability on the M3 host

The provisioning receipt (2026-07-12 20:53Z) and the base-inference receipt
(2026-07-12 21:09Z) record that the F16 control was not yet on the M3 host,
because transfer from the artifact host was rejected by SSH authentication.
The F16 control was provisioned afterwards: the F16/Q4 comparison receipt
(recorded 2026-07-13 00:47Z) verified its manifest, `origin.json`, all 267 shard
filenames and total bytes, loaded it with shard-hash verification on the
24 GB M3, and ran it. Every later receipt uses the F16 control on that host.

## Fixed identities

| Field | Mixed Q4 (`qwen-3-5-9b-q4k-ehaf16`) | F16 control (`qwen-3-5-9b-f16-af32`) |
| --- | --- | --- |
| Source | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | same |
| Manifest SHA-256 | `ac1eff7f371b7d676a5b9a7ddcf94e2d01d3e1043f499dede2fc41249ef55956` | `2ff1f1eec0345bd379614b8b0bafd66957bbec1c22072132661a6e35300ad398` |
| Origin SHA-256 | not recorded | `0099b37f79a12bc2663adb7f4851852d2aa773f03bd892d34ec91b18aae97c3b` |
| Shard-set hash | `sha256:27c1ba6d379e43eff7eae12b6ccc83081fe5248d1739e7877e82428d52daf228` | `sha256:7d4053a21a45d06690a1e4ac3377cd4d2d46a7239ce94f739bb2e5830f16900d` |
| Weight-pack hash | `sha256:dfeea5dc362804bd197d1f96effea74c4cd803eb93b8304c633057f4f172f237` | `sha256:d337596cd6ae6cb360b60bca56d95b2e24f5253113953455b6565ffcd055ee46` |
| Shards / bytes | 132 / 8,824,455,680 | 267 / 17,907,606,528 |
| Conversion config | `src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json` | `src/config/conversion/qwen3/qwen-3-5-9b-f16-af32.json` |

## Fixed prompt

- Prompt: `Answer with exactly the WGSL scalar type name for a 32-bit floating-point value.`
- Input: 31 tokens; the IDs are in each machine-readable receipt.
- Runtime profile: `profiles/qwen-3-5-9b-metal-correctness`
- Sampling: greedy, batch size 1, 16 maximum generated tokens, F16 KV cache,
  no speculative decoding, and no performance overrides.

## Chronology

| Date | Receipt | Result at the time |
| --- | --- | --- |
| 2026-07-11 | [RDRR receipt](qwen35-9b-rdrr-2026-07-11.md) | F16 artifact accepted (`` `f32` `` on WebGPU/Vulkan); three Q4_K lanes, including the later mixed-Q4 artifact, stopped at `<|im_end|>` and were not published. |
| 2026-07-12 | [M3 provisioning](qwen35-9b-metal-provisioning-2026-07-12.md) | Mixed-Q4 artifact provisioned on the M3 with exact identity; F16 control not yet on the host. |
| 2026-07-12 | [M3 base inference](qwen35-9b-metal-base-inference-2026-07-12.md) | Mixed Q4 reproducibly selects `<|im_end|>` on Metal; F16 control still pending. |
| 2026-07-12 | [M3 F16/Q4 comparison](qwen35-9b-metal-f16-q4-comparison-2026-07-12.md) | F16 control runs on the M3 and generates `` `f32` ``; Q4 stops; first material divergence at `layer.0.ffn.gate`. |
| 2026-07-12 | [Layer-0 Q4K oracle](qwen35-9b-metal-layer0-q4k-oracle-2026-07-12.md) | Q4 Metal gate kernel matches a scalar Q4_K reference; the layer-0 difference is quantization loss, not a decode or kernel error. |
| 2026-07-12 | [Native-training parity design](qwen35-9b-doppler-native-training-parity-design-2026-07-12.md) | Training-parity design; treats the mixed-Q4 artifact as out of scope. |
| 2026-07-13 | [Architecture contract correction](qwen35-architecture-contract-correction-2026-07-13.md) | Native-training graph order fixed; the suspected inference-manifest norm defect was disproved. |
| 2026-07-13 | [Gated SiLU correction](gated-silu-runtime-correction-2026-07-13.md) | Split gated SiLU defect fixed in source; released in 0.4.9 on 2026-07-14. |
| 2026-07-14 | [0.4.9 base parity](qwen35-9b-metal-049-base-parity-2026-07-14.md) | Mixed Q4 and F16 both generate `f32`; the Q4 rejection is reversed (current verdict). |

## Retained evidence

The 2026-07-12 M3 receipts and raw logits are tracked under
`reports/qwen-3-5-9b-metal/2026-07-12/`. The 2026-07-14 raw receipts
(`reports/qwen-3-5-9b-metal/2026-07-14/`) and the 2026-07-11 inference receipt
(`reports/qwen-3-5-9b-f16-af32/2026-07-12T02-21-24.974Z.json`) are not retained
in the tree; their SHA-256 values remain in the dated JSON receipts.
