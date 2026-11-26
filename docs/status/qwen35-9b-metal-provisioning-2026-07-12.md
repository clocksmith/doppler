# Qwen 3.5 9B M3 provisioning receipt

> Superseded by [qwen35-9b-metal-status.md](qwen35-9b-metal-status.md) (2026-07-14 reversal of the mixed-Q4 rejection).

The pinned mixed-Q4_K candidate is provisioned locally with exact content
identity. The accepted F16 control and V11 GRPO adapter are not yet provisioned
on this host because the AMD machine still rejects SSH authentication. The F16
control was provisioned later the same day; see
[F16 control availability](qwen35-9b-metal-status.md#f16-control-availability-on-the-m3-host).

Machine-readable evidence is in
[`qwen35-9b-metal-provisioning-2026-07-12.json`](./qwen35-9b-metal-provisioning-2026-07-12.json).

## Exact local artifact

- model: `qwen-3-5-9b-q4k-ehaf16`
- source revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- shards: `132`
- shard bytes: `8,824,455,680`
- shard-set and weight-pack hashes: see [canonical status](qwen35-9b-metal-status.md#fixed-identities)
- conversion config: `src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json`

The matching conversion requires all six recurrent exclusions:

- `linear_attn.conv1d.weight`
- `linear_attn.in_proj_a.weight`
- `linear_attn.in_proj_b.weight`
- `linear_attn.in_proj_qkv.weight`
- `linear_attn.in_proj_z.weight`
- `linear_attn.out_proj.weight`

## Preserved negative evidence

Three earlier outcomes remain in the JSON receipt:

1. The M3 F16 reconstruction matched the pinned tensor count and byte size but
   not its shard-set or weight-pack identity, so it is rejected as the control.
2. The first Q4 reconstruction exactly reproduced the checked-in 97-shard
   all-Q4 rejected ablation.
3. Keeping only the three large recurrent projections in F16 produced 132
   shards but was exactly `10,174,464` bytes short. That deficit identified the
   three original recurrent exclusions that also had to remain F16.

No failed conversion was relabeled as accepted, and no artifact was selected by
size alone.

## Claim boundary

This receipt proves source and mixed-Q4 artifact identity only. It does not prove
base-model inference correctness, adapter inference correctness, runtime
performance, or any training improvement. Doppler V12 training was not run or
modified.
