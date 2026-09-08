# W4A16 feed-forward metadata

The pinned Gemma conversion passes shard integrity and retains valid W4A16
weight descriptors. `baseline-error.json` records generation failing because
the feed-forward fusion planner rejected that known dtype as missing metadata.
`diagnostic.json` reproduces the rejecting validator independently of model math.

The validator now accepts the declared W4A16 descriptor. Existing fusion policy
still excludes it from native fused F16 execution, allowing the manifest's
separate packed matrix operations. Unknown or absent metadata remains an error.
The focused regression checks both admission and fusion rejection:

```sh
node tests/inference/ffn-execution-v1-precision-contract.test.js
```

`validation.json` identifies the correction and passing repository checks. Unused
imports in the touched module were removed; package limits remain unchanged.
`candidate-error.json` records the subsequent physical model run advancing to
logits execution. Further tracing found finite prefill logits followed by device
loss during decoding; `remaining-device-loss.json` retains that distinction.
This correction does not qualify complete generation, answer quality, or speed.

`evidence.json` hashes the retained observations and names their original local
paths. The full development command configurations and traces remain in the
Reploid local-journeys workspace named by those paths. Source-model inspection
and converted shard identities are retained here; this directory alone is not a
standalone model distribution bundle or independent-operator reproduction.

Component: text feed-forward execution planning.
Intent: preserved.
Acceptance evidence: focused precision-contract regression, full repository
checks, descriptor diagnostic, and physical before/after failure boundaries.
Boundary effects: no new format, kernel, manifest field, or fusion policy.
