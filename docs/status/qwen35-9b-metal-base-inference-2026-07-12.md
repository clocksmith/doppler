# Qwen 3.5 9B M3 base-inference status

The exact 8.8 GB mixed-Q4_K candidate reproducibly fails the deterministic
first-token check on the Apple M3 Metal lane. Both runs produced the same raw
logits SHA-256, selected `<|im_end|>` (`248046`), and stopped with an empty
answer. This is base-model inference evidence only; it is not a training,
adapter, or performance result.

## Fixed inputs

- Doppler commit: `f1ecec81f84d31ffc9a66645d46ebefe820fa6b7`
- Source checkpoint: `Qwen/Qwen3.5-9B` at
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Q4 manifest SHA-256:
  `ac1eff7f371b7d676a5b9a7ddcf94e2d01d3e1043f499dede2fc41249ef55956`
- Q4 shard-set hash:
  `sha256:27c1ba6d379e43eff7eae12b6ccc83081fe5248d1739e7877e82428d52daf228`
- Q4 weight-pack hash:
  `sha256:dfeea5dc362804bd197d1f96effea74c4cd803eb93b8304c633057f4f172f237`
- Runtime profile: `profiles/qwen-3-5-9b-metal-correctness`
- Prompt: `Answer with exactly the WGSL scalar type name for a 32-bit floating-point value.`
- Sampling: greedy, batch size 1, 16 maximum generated tokens, F16 KV cache,
  no speculative decoding, and no performance overrides

The complete 31-token input sequence is recorded in the machine-readable
receipt. The selected token's logit was `17.236438751220703`; the runner also
captured the top 20 candidates and all 248,320 raw F32 logits. The raw vector
is finite and has SHA-256
`cb3420f9986a2f2e81fdfe204064d8aec24dffa8488e6bcdb9c795092a1f6162`.

## Boundary evidence

The diagnostic replay recorded 469 ordered slices from `embed.out` through
`logits.final`. Every captured slice reports finite statistics. This trace
does not by itself identify a bad boundary: the exact F16 control must be run
with the same input and capture plan before a first numerical divergence can
be named. Q4_K scale/min decoding, recurrent projection materialization, and
mixed-precision boundary handling remain hypotheses, not conclusions.

## Gated work

The exact accepted F16 control is not yet present on this host. Transfer from
the authorized artifact host is currently rejected by SSH authentication, so
no load or allocation outcome is claimed. Adapter activation is also not run:
the V11 GRPO adapter test remains gated on coherent base-Q4 token parity.
Dispatch, attention, and kernel-speed optimization were not started.

## Evidence

- Machine-readable status:
  [qwen35-9b-metal-base-inference-2026-07-12.json](qwen35-9b-metal-base-inference-2026-07-12.json)
- Base receipt:
  `reports/qwen-3-5-9b-metal/2026-07-12/q4-base-first-token.json`
- Base raw logits:
  `reports/qwen-3-5-9b-metal/2026-07-12/q4-base-first-token-logits.f32`
- Boundary receipt:
  `reports/qwen-3-5-9b-metal/2026-07-12/q4-boundaries.json`
- Boundary raw logits:
  `reports/qwen-3-5-9b-metal/2026-07-12/q4-boundaries-logits.f32`

The JSON status records host and Metal metadata, artifact identities, runtime
configuration, input tokens, top-k logits, exact evidence hashes, and both
commands verbatim.
