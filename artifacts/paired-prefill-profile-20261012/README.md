# Immediate paired prefill profiling diagnostic

Status: CPU controls pass. No physical profiling run has started. This instrumentation does not qualify a release or explain the 300-second failure yet.

The retained original run is Reploid `artifacts/concurrent-memory-observation-20261012/bounded-result/`. Mac group0 accepted the original1588-token prompt but produced no activation before the fixed300000ms deadline. It submitted459 command buffers, had no pending pipeline creation or device loss, then confirmed its cleanup queue barrier. Host gaps following layer-boundary submissions are not GPU kernel timings.

## First broken observation boundary

`partition-execution.js` consumes the existing immediate-prefill policy and therefore passes no recorder to model kernels. `kernel-execution.js` calls `dispatch.js`, which creates and submits one encoder per operation. The layer recorder previously profiled only its own carry copies and discarded timing results for immediate layers. Setting the existing profiler flag alone therefore could not identify the GPU work responsible for the delay.

The correction explicitly leases the isolated device's immediate dispatches to the existing `CommandRecorder`. Every original kernel still gets one `submitPartial()` without a host wait. Uniform construction and ownership retain their immediate path. Existing layer-completion barriers resolve timestamp queries, emit per-pipeline/workgroup timings through the existing Profile logger, and rotate the recorder. The lease rejects overlap or unavailable timestamp support and ends before the final LM-head operation. It is a single-resident/device diagnostic scope, not a concurrent executor profiler.

The timestamp reader's strict observation mode rejects incomplete query coverage, blocked readback, and reversed timestamps. It preserves valid durations over60seconds rather than silently dropping them. Submitted abort now also settles query-resource cleanup if completion failed before timing resolution. Scalar/profile-disabled behavior stays on the original dispatch path.

## Proposed bounded physical operation

Use the existing immutable concurrent fixture with only the separately bound instrumented package and profiler-enabled configuration. Replay the same original1588 baseline, direction0, Mac group0 then AMD group1. Preserve the fixed300000ms operation,10000ms cleanup,2000ms observation bounds, original inputs/generation,1.42B per executor, unchanged arithmetic and F16KV. The first failure was on Mac, so an AMD-only replacement cannot attribute it.

Use a fresh owned Mac controller and AMD peer context after root confirms the reservation and a live process check. Public profiles/contributors remain untouched. Root currently holds launch while investigating a public conversation failure. No physical command is authorized by this document alone.

Retain each completed layer/token-chunk Profile record, observer submission and preparation events, and the primary deadline/cleanup result. Compare summed GPU pass durations with encode wall and barrier wait; encoding and GPU work can overlap, so their sum is not an exclusive CPU/GPU decomposition. Profiling introduces query buffers and one resolve submission per boundary; record that cost and shared-machine interference. Missing timestamp coverage rejects the observation instead of becoming zero time.

Relevant arithmetic, unchanged by this diagnostic: `matmul_paired.wgsl` (one output/workgroup,64-lane reduction), `gated_delta_conv_paired.wgsl`, `gated_delta_recurrent_paired.wgsl`, `attention_paired.wgsl`, paired normalization and pointwise kernels. The original FFN projection dispatch included[65535,15,1] workgroups at256 prompt rows. Large dispatch geometry is a hypothesis for measurement, not established causation.

## CPU evidence

- `tests/gpu/immediate-dispatch-profile.test.js`: binding/geometry identity, immediate submission count/order, rotation, overlap, missing timestamps, failed submit and release.
- `tests/gpu/command-recorder-cleanup.test.js`: actual recorder ports, timestamp pass descriptors, no per-kernel queue barrier, map/queue/capacity failures, query destruction, valid61-second observation.
- `tests/inference/partition-prefill-lifetime.test.js`: actual resident orchestration keeps immediate layers,12 kernel submissions and3 existing layer barriers, aggregates timing, and releases the lease/resources on failure.
- Source type, style, architecture, dependency and export checks accompany the scoped checkpoint.

Component: `doppler.runtime-source.gpu` and `doppler.runtime-source.inference.pipelines.text`. Intent: preserved. Boundary effects: explicit profiling observation across immediate dispatch; no model policy, graph, precision or arithmetic change.
