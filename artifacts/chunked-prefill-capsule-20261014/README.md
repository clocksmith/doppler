# Signed Capsule: candidate 2df37d88 with prefill chunked at 16 tokens

Component: signed model delivery (local qualification scope). Intent preserved. Boundary effects:
a new Capsule and one new release event in the test authority's history. No archive, catalog,
installed package or public pin changed.

## Why

A 267-token prompt hung the Intel GPU in the three-stage application route. i915 reset the render
engine after a dispatch ran past its 7,500 ms preemption timeout. The earlier candidate Capsule
(`67f240af...`) declares `prefillTokenChunkSize` 256, so a 256-row prefill chunk ran in single
dispatches on intel-host. This Capsule declares 16, which bounds every Intel prefill dispatch to
about 2 s at the measured per-row cost. See Reploid
`artifacts/phase-timing-20261014` and `artifacts/intel-gpu-reset-20261014`.

## What it binds

[`build.js`](build.js) follows `../kernel-completion-repair-20261013/signed-capsule/build.js` step
for step: same runtime archive `2df37d88...`, source export `a5f3ca4c`, manifest `6b2cb026...`,
weight and tokenizer pieces (asserted equal to the previous signed index), release policy, and
test-scope signer `reploid-paired-qualification-20261012`. What differs:

- **Reports:** qualification reports and contract come from Reploid
  `artifacts/completion-qualification-3-20261014/b`. All 55 prefixes pass within 0.001 (maximum
  error `9.520044947919359e-07`), with prefill chunked at 16 tokens.
- **Partition order:** mac-host browser stage 0, intel-host native stage 1, amd-host native
  stage 2. The output head is on AMD.
- **Initial execution identity:** it declares `prefillTokenChunkSize` 16, as observed live
  (execution id `sha256:c4819f17...`).
- **Generation report:** it is rebuilt from that capture's first request. The 19-token prompt is
  now prefilled as chunks of 16 and 3, and it still reproduces the independent source tokens
  exactly.

## One placement per Capsule

The Capsule format can hold several complete placement groups, since
`assertOrderedResidentQualificationSet` groups records by contract. The rig at `a5f3ca4c`, though,
admits reports against a single resident contract file (`residentQualificationContractPath`). So
this Capsule binds placement B only. Its release exclusions name that partition order; other role
assignments are not qualified by it. Placement A (output head on Intel) passed the same campaign
(`completion-qualification-3-20261014/a2`). A Capsule bound to A can be built from that evidence
with `node build.js <output> a2` as a later release event.

## Identity and release

- **Capsule v3:** semantic root `sha256:171132d428ce64eaf706c637ee22fea0103f50a21e39e9899cac5d91b5f94234`,
  envelope `sha256:3b4a3e3d308be8fd5e9d0c9fe2cae7a6d06398d2a859b3f00b130225a017bd95`.
- **Release history:** [`release-events.json`](release-events.json) continues the authority's
  history: sequences 1 and 2, the sequence 3 renewal (`74ab985f...`), then this Capsule at
  sequence 4.
  - Digest `sha256:7f62c19a88165fded22d857860da729cb82f02a69f30e6f48765f35f1195ee45`.
  - Expires 2026-10-15T13:21:16Z, failing closed.
  - Verified from the exact 0.6.28 archive for a host at checkpoint 3 and for a host with no
    history ([`exact-runtime-verification.json`](exact-runtime-verification.json)).
  - Becoming eligible supersedes the sequence-3 Capsule for hosts that accept this history.

## Not established

- Physical signed opening of this Capsule.
- The signed catalog and staged application packages.
- Installed conversations.
- Placement A under this Capsule.
- Public eligibility. Production signing remains the public-promotion gate.

The private signing key is read from its file and is not in this repository.
