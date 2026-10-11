# Committed runtime revalidation

The existing physical seeded-batch regression passes against immutable commit
`d7db1a35ee35bbe96f1162867d03782753aa89f0`, after the reasoning-output and completion
contract changes were combined. No candidate source overlay was used.

```sh
node tests/integration/seeded-batch-physical.test.js \
  reports/local/seeded-batch-physical-20261011/combined-runtime/physical.json
```

Chrome 146.0.7680.177 used AMD Radeon 8060S Graphics, RADV 26.0.3 and Vulkan
1.4.335. The raw receipt records browser/driver identity and every served source
and shader digest. Fixed embedding/logit fixtures replace model computation;
the sampler, recorded batches, readback, output delivery and cleanup execute
through the actual committed implementation.

All five seeded whole/split/same-index retry controls match the independent
expected draws. Partial-batch cancellation emits no later token and retains
`aborted`. A fresh attempt succeeds. `onToken` and `onBatch` exceptions preserve
their identities without triggering GPU fallback. Every observed boundary has
zero active and deferred-cleanup bytes; reusable pool storage remains bounded at
18,432 bytes. No uncaptured GPU errors occurred. The harness closed its browser
and owned resources after completion.

Earlier failures remain in the parent evidence directory. This establishes the
scoped sampler/lifecycle regression after integration, not model quality,
numerical qualification, a promoted archive or distributed-chat acceptance.
No runtime, model, standard archive, tolerance or reference changed in this run.

Component: `doppler.tests`. Intent: preserved. Boundary effects: evidence only.
