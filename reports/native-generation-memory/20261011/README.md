# Sampled-logits ownership in installed Node generation

The text generator discarded an owned GPU logits buffer after consuming its CPU
readback. Repeated answers and cancellation therefore accumulated active buffers.
Replay-prefill decode duplicated the same sampling path.

The generator now releases the buffer in `finally`, including sampling failure,
and replay decode delegates to that owner. Advanced decode and GPU-only return
paths continue to transfer their buffers to the caller. Model bytes, precision,
context, tensor operations, sampling policy and GPU pool budgets are unchanged.

The regression test reproduces the missing release on the pinned parent and
covers success, sampling failure, decode failure and successful reuse. Source
architecture/style checks and the complete unit suite pass. Exact baseline and
candidate npm archives also pass installed Node generation through explicit Doe
and Dawn Vulkan providers, matching the independent CPU token/text reference.
Cancellation settles submitted queues before the next correct answer and cleanup.

The [retained Doe-hosted comparison](https://github.com/doerun/doe/tree/main/reports/maintenance/20261011-doppler-generation-memory)
owns policy, identities, all raw receipts, independent reference, allocation
observations, timing results and failed checks. Active buffer capacity returns to
its loaded state on both providers without evicting compatible weights.

This is a proven resource-lifecycle correction. It does not establish a material
peak process-RAM advantage: that frozen primary check fails. Foreign Chrome GPU
activity makes the cohort timings ineligible, and a Doe load-latency bound also
fails. Keep the unsuccessful results and existing thresholds. No package is
published or deployed, no Reploid numerical tolerance changes, and no native
browser-contribution or unrelated-application memory advantage is qualified.

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved.
Acceptance evidence: `tests/inference/decode-sampling-ownership.test.js`, existing
decode-output/GPU-transfer controls, `npm run test:unit`,
`npm run source:architecture:check`, `npm run source:style:check`, and linked
physical receipts. Boundary effects: owned CPU-sampled logits cleanup; public
interfaces and application coordination remain unchanged.
