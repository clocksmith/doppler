# Community LoRA round qualification — 2026-10-17

Component: `doppler.runtime-source.experimental`. Intent: preserved.
Boundary effects: the experimental training API gains checkpoint-bound gradient
rounds and versioned export metadata; applications still own transport,
placement, deadlines, publication and evaluation. No new network service,
Reploid training UI, or model-support promotion is introduced.

## Physically executed

Exact base: `qwen-3-5-0-8b-q4k-ehaf16`, final layer 23 `down_proj`, frozen Q4_K
base, rank 2, alpha 4, f32 adapter parameters and gradients. Both participants
verified base-manifest hash
`sha256:cf20c690cc397a4bd8e64ee4b00158114947ee75568c0ae8f2df8a9ebc052965`.

| Participant | Runtime and device | Supervised tokens | Local loss |
| --- | --- | ---: | ---: |
| M3 | Node/WebGPU, Apple M3 Metal, macOS 26.6.2 | 3 | 9.030719757 |
| AMD | Node/WebGPU, Radeon 8060S Vulkan/RADV Mesa 26.0.3 | 1 | 12.966977119 |

Both used the same [round checkpoint](round.json). Retained gradient contributions:
[M3](contribution-m3.json), [AMD](contribution-amd.json). These are actual
full-model frozen-base forwards and native LoRA backwards on separate computers.
The small token-ID samples are qualification inputs, not a useful training corpus.

The [coordinator receipt](applied.json) records one AdamW step over four
supervised tokens, weighted loss 10.014784098 and unclipped gradient norm
0.060910473. Participants did not independently update and average adapters.
The optimizer ran once on the M3 after receiving both gradient payloads.

[Exported adapter](adapter/adapter.json), version `0.0.1`, and
[weights](adapter/adapter.safetensors) have SHA-256
`5fe5d81fe9d274ceb01a55705be69b8645f0fcda5689eea7cbd5a8c912cc8a7f`.
A fresh coordinator restored the same checkpoint and reproduced those exact
bytes after the final method extraction. Per-action receipts retain actual
source hashes; they do not pretend every action used a single source revision.

The [independent reload receipt](reloaded.json) verifies checksum rejection of
corrupted weights, version/identity loading, finite logits, nonzero adapter
effect (max absolute logit delta 0.000169039), and exact base restoration
(delta 0). This is execution and isolation evidence, not improved task quality.
The adapter has not been published as ready or entered a public community catalog.

## Numerical and failure gates

The real GPU suffix oracle compares an unequal 2:1 contribution split against
one combined-batch optimizer update, including Adam moments and clipping:

- [Metal oracle](metal-suffix-oracle.log): max absolute error below 1e-5.
- [Vulkan oracle](amd-suffix-oracle.log): max absolute error below 1e-5.
- Missing participants, duplicate contributions, altered round/sample identity,
  stale checkpoints, non-finite gradients and replayed completed rounds fail.
- Malformed checkpoint tensors are all validated before parameter writes.
- Invalid rounds leave the coordinator checkpoint unchanged.
- Concurrent trainer operations fail explicitly; optimizer failure makes the
  trainer unavailable rather than silently continuing with possibly partial state.

Oracle fixtures are not whole-model quality references. Cross-device gradient
production does not establish speedup, arbitrary layer support, adversarial
contributor trust, or fault-tolerant distributed optimization.

## Reproduce with provisioned models

Each command performs at most one round action. The retained execution wrappers
used a 300-second process timeout; the GPU oracle used 90 seconds on AMD.

```sh
node tools/qualify-community-lora.js --config reports/community-lora-20261017/prepare.json
node tools/qualify-community-lora.js --config reports/community-lora-20261017/contribute-m3.json
# On AMD: copy round.json and contribute-amd.json into the snapshot working directory.
node tools/qualify-community-lora.js --config contribute-amd.json
# Copy contribution-amd.json back into the report directory, then:
node tools/qualify-community-lora.js --config reports/community-lora-20261017/apply.json
node tools/qualify-community-lora.js --config reports/community-lora-20261017/verify.json
node tests/training/native-lora-round-gpu.test.js
```

Use the absolute local model paths in the JSON contracts or supply an already
provisioned equivalent. The AMD source snapshot used the host's existing Node
WebGPU dependency and separate files under `.tmp/community-lora-20261017`.
Public contributors were not restarted. The two contribution actions did not
run simultaneously; this establishes one shared optimizer round across devices,
not concurrent speedup or sustained multi-round service.

The initial reload probe incorrectly expected the generation callback to expose
first-token logits. Its [failure log](reload-missing-callback.log) is retained.
The corrected probe uses the existing explicit `prefillWithLogits` API; no
runtime fallback was added to conceal the harness failure.

## Development checks and remaining work

Passed: protocol regression test, Metal/Vulkan GPU suffix oracle,
LoRA safetensors export test, request-timing and batching harness regression tests,
source/declaration type checks, export parity, unused-local and source-style checks.
The benchmark request-timing extraction preserves the preceding audit's measured
semantics and removes unused imports; its new implementation is strictly checked.

The source architecture check still fails on the untouched attention executor:
971 lines against its existing reviewed ceiling of 960. Earlier skipped CI and
fake-tokenizer/full-suite failures remain open. No full release qualification
is claimed by these focused checks.

Still required for a community product: evaluated training datasets, repeated
rounds with contributor departure and explicit retry, application integration
for job placement and publication, and user retention. Native training is still
bounded to the named 0.8B final-layer target; 2B and other targets remain outside
this native qualification. Performance optimization remains open: the current
M3 comparison still loses warm first-token latency and token-delivery tails to
Transformers.js. See [the timing audit](../warm-fairness-20261017/README.md).

## Packed application check

The [local npm candidate](package/doppler-gpu-0.6.31.tgz) retains version 0.6.31
and is **not published**. Its [archive identity](package/identity.json) is
SHA-256 `111174597255a8ffe93c98354bb266e0a4ce26f4df388ec97e9494702fb52da6`.
A [separate application probe](package/probe.js) imported only the public
`doppler-gpu/training` entry from the extracted archive, restored the common
checkpoint, applied both device contributions and exported the identical adapter
weight digest. [Application receipt](package/application.json) and
[log](package/application.log) are retained. Existing host WebGPU dependencies
and provisioned model files were used; this is not a clean-machine installation
or a new npm release. API export inventory and public declaration parity pass.
