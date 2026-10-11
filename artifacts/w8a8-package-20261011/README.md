# W8A8 implementation package handoff

The installed test candidate is **not a replacement for Reploid's accepted
package**. W8A8 operator correctness and the repaired generation lifecycle pass
their scoped token-callback cancellation checks. A stronger abort immediately
after GPU submission exposes pending retirement on prefill rejection, so complete
cancellation settlement remains open for this candidate. Gemma model quality and
the separate distributed numerical gate also remain open. No Mac execution or
Reploid deployment is claimed here.

## Exact identities

| Item | Identity |
| --- | --- |
| Accepted standard archive, unchanged | `9280ead598b098f9f3ad49574a001e88c3ad4b1a433d58349e79ee0e9c01fccc` |
| Installed test candidate | `79b7c68ceb23d3e156e5e6287d978bfbe218940b832c2460b80efe2472e9b44b` |
| Filename and version | `doppler-gpu-0.6.27.tgz`, `0.6.27` |
| Candidate source | `86e3761beeaf4dd702acf813a9adf7e0de7ac1ee` plus the recorded unshipped tooling state |
| Packed / unpacked bytes | 3,366,814 / 13,213,186 |
| Package members | 1,917 |
| Gemma W8A8 manifest | `6bd380101b72d79aa05c23d51783a9faab9505808d63c4bd86520e95d255c03c` |
| Frozen W8A8 acceptance fixture | `890a28e543ef614bcfb0fa60de195c189d30c1216f29804559a2b2de7b8edd5a` |

The candidate resides at
`/var/tmp/doppler-w8-installed-settled-20261011/doppler-gpu-0.6.27.tgz`.
`settled-package-inventory.json.gz` records its package inventory. The adjacent
`settled-installed-*` files retain the isolated installation receipt, commands,
source state and their uncompressed digests. Browser qualification serves the
installed archive's files; it does not silently substitute checkout source.
This is local test delivery, not a published archive or a refreshed Reploid bundle.

Earlier candidate identities and failed physical invocations remain retained.
Version equality alone does not authorize substitution.

## Operations and boundaries

Existing public operations remain `generate.v1`, `embed.v1`, `rerank.v1`,
`scoreChoices.v1` and `encodeSequence.v1`. Availability depends on the admitted
model and operation contract; this batch does not qualify every operation for
Gemma or every backend. Reploid must consume those declarations and returned
results rather than reproduce model mathematics.

- Generation retains model-owned tokenization, sampling, declared stopping and
  seeded behavior. Public completion distinguishes `eos-token`, `stop-token`,
  `stop-sequence` and `max-tokens`; cancellation and failure remain errors where
  the public contract requires them. Internal pipeline `aborted` metadata is
  not a successful natural completion. Missing legacy completion metadata is
  represented as unknown, not inferred from text.
- Embedding subdivision applies only to the existing qualified complete-text
  application, with independent per-input state and original ordering. It does
  not make token fragments, generation or list-dependent reranking separable.
- Choice scoring returns all contextual label logits and their token identities.
  These are uncalibrated model scores, not permission or general truth scores.
- Resident partition recovery continues to advertise no input replay, checkpoint
  export or checkpoint import. A new attempt is restart, not restored live state.
- Cancellation waits for owned submitted work and cleanup; it cannot interrupt
  an already executing GPU command. The allocation ceiling remains
  **1,420,000,000 bytes**, distinct from physical device memory.

The W8A8 ABI declares symmetric codes [-127,127], per-output-channel weight
scales, per-token activation scales, exact bounded I32 accumulation and two
explicit F32 rescale stages. K is bounded by 133,144. Invalid nonfinite/subnormal
inputs, forbidden codes, padding, scales and overflow reject. Packed-dot
selection requires `packed_4x8_integer_dot_product`; feature presence alone
does not establish acceleration. Embeddings, normalization, attention, caches
and residuals retain their declared precision.

## Evidence and remaining gates

- [Conversion](../w8a8-conversion-20261011/README.md): 126 W8A8 matrices,
  110 F16 tensors and seven checked shards.
- [Physical operators](../w8a8-kernels-20261011/README.md): portable and packed
  variants match independent activation bytes, accumulators and output bits for
  every frozen projection shape and boundary controls.
- [Installed model results](../w8a8-model-qualification-20261011/README.md):
  generation, repeated use, submitted cancellation, recovery and settled unload
  are checked separately from failed task quality and likelihood criteria.
  Candidate `79b7c68` peaks at 668,488,920 accounted allocation bytes; sampled
  renderer RSS is 926,752,768 bytes. Neither measure is physical VRAM usage.
  Later packed-model controls abort directly after `queue.submit`: active
  ownership returns to its baseline, but 13,094,144 bytes across 133 deferred
  resources remain after rejection and queue completion. Unload clears them;
  this does not meet the stronger cancellation-settlement requirement.
- [Embedding regression](../w8-runtime-embedding-20261011/README.md): all 18
  complete texts, ordering, submitted cancellation and reuse passed on candidate
  `975adf34`; that report is not relabeled as a new `79b7c68` physical run.
- [Scoring regression](../w8-runtime-scoring-20261011/README.md): candidate
  `79b7c68` passes all 12 selections and all 24 returned logits against the
  retained independent reference (maximum error 0.012686491, unchanged limit
  0.05), plus submitted cancellation, settled reuse and unload on Node/Vulkan.
  Its operation-specific tolerance does not alter the distributed 0.001 gate.
- [Repository checks](../w8a8-repository-checks-20261011/README.md): 917 unit files,
  36 Capsule files and repaired generated checks. This is composite evidence,
  not an immutable-source `check:green` invocation.
- [Distributed reference disposition](../whole-model-f64-complete-20261011/README.md):
  independent F32 and F64 evaluations do not qualify a replacement. Historical
  failures and the 0.001 threshold remain unchanged; captured outputs are not
  activated as an accuracy reference.

The frozen Gemma application criterion requires 10/12 tasks and preservation of
F16-passing tasks. The independent W8A8 source achieves 5/12; the installed GPU
path achieves 4/12 and also fails the frozen F16-relative likelihood allowance.
Correctness diagnostics, portable/packed full-model comparisons and separately
reported prefill/decode measurements must retain these failures. No quality,
capacity or speed advantage is established by publication or operator agreement.

Component: `doppler`. Intent: preserved. Acceptance evidence: linked scoped
reports and exact installed receipt. Boundary effects: public consumers retain
the accepted archive; Reploid application integration remains consumer-owned.
