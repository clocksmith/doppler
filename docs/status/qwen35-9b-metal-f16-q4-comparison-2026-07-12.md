# Qwen 3.5 9B M3 F16/Q4 correctness comparison

> Superseded by [qwen35-9b-metal-status.md](qwen35-9b-metal-status.md) (2026-07-14 reversal of the mixed-Q4 rejection).

> **Later audit (2026-07-13):** an initial hypothesis that the v1 manifest used
> the wrong post-attention normalization route was disproved. The v1 inference
> graph already has the required norm order and split-half RoPE pairing. This
> receipt retains its original one-prompt F16/Q4 comparison scope. See
> [the native-training architecture correction](qwen35-architecture-contract-correction-2026-07-13.md).

The exact accepted F16 control fits and runs on the 24 GB Apple M3. With the
same prompt, input tokens, correctness profile, and greedy sampling used for
the mixed-Q4 candidate, F16 generated `` `f32` `` and Q4 stopped immediately
at `<|im_end|>`.

## Fixed artifact identities

| Field | F16 control | Mixed Q4 candidate |
| --- | --- | --- |
| Source revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | same |
| Manifest, shard-set and weight-pack hashes | See [canonical status](qwen35-9b-metal-status.md#fixed-identities) | same |
| Shards / bytes | 267 / 17,907,606,528 | 132 / 8,824,455,680 |

The F16 manifest and `origin.json` match their pinned SHA-256 identities. All
267 expected shard filenames and total stored bytes were present before the
run. Doppler loaded with shard-hash verification enabled.

## Base result

The identical 31-token input produced:

| Artifact | First token | First-token logit | Output |
| --- | --- | ---: | --- |
| F16 | backtick (`63`) | 17.450632095336914 | `` `f32` `` |
| Q4 | `<|im_end|>` (`248046`) | 17.236438751220703 | empty |

F16 generated `63, 69, 18, 17, 63, 248046`. Q4 generated only `248046`.
Every one of the 248,320 captured logits was finite in both runs.

## Ordered boundary comparison

F16 generated six tokens, so its diagnostic receipt contains 2,489 records.
Q4 stopped after one token and contains 469. The comparable F16 prefix is its
first 469 records; both sequences end at `logits.final` and have identical
operator order.

- `embed.out` at index 0 matches exactly.
- The first nonzero sampled difference is index 1,
  `layer.0.attn.qkv_proj`: maximum absolute difference
  `0.0000002384185791015625`. It is within the configured projection limit.
- The first configured tolerance violation is index 7,
  `layer.0.attn.post_attn`: maximum absolute difference
  `0.000002216547727584839`.
- The first material change—defined in this receipt as an absolute sampled
  difference above `0.001`—is index 9, `layer.0.ffn.gate`: maximum absolute
  difference `0.018221039324998856`, maximum relative difference
  `0.5089205363850464`.

The index-9 weight is
`model.language_model.layers.0.mlp.gate_proj.weight`. It is F16 in the control
and Q4_K_M in the compressed candidate. Recurrent QKV and attention-output
weights before it are F16 in both artifacts. This makes the layer-0 feed-forward
gate the first material mixed-precision boundary in this comparison.

That finding does not yet distinguish ordinary quantization loss from a Q4_K
scale/min decoding or materialization error. The next correctness diagnostic is
to compare this projection against a trusted dequantized/reference calculation
using the same `layer.0.ffn.in` values. No runtime fix is claimed yet.

## Capture-mode control

The F16 base and snapshot-enabled runs chose the same tokens and output, but
their complete raw-logit vectors were not byte-identical. Their maximum logit
difference was `0.32733678817749023` with RMSE `0.06263648535794095`.
Therefore the localization compares snapshot-enabled F16 against
snapshot-enabled Q4 and does not treat the first microscopic exact difference
as the root cause.

## Evidence and claim boundary

The machine-readable receipt is
[qwen35-9b-metal-f16-q4-comparison-2026-07-12.json](qwen35-9b-metal-f16-q4-comparison-2026-07-12.json).
Raw receipts and logits are under
`reports/qwen-3-5-9b-metal/2026-07-12/`.

This establishes M3 base-model correctness for the exact F16 artifact and a
reproducible base-model failure for the exact mixed-Q4 artifact under the
runtime of that date. Doppler 0.4.9 reversed the Q4 failure; see the
[canonical status](qwen35-9b-metal-status.md). It does not
claim adapter correctness, runtime performance, or any new training result.
The frozen V12 AMD/ROCm training matrix was not changed or rerun.
