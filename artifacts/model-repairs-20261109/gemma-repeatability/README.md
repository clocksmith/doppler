# Gemma F16 repeatability repair

The repeated continuation mismatch was a synchronization defect in
`rmsnorm_small_f16`, rather than an unavoidable F16 tradeoff. After reducing a
row's maximum magnitude, the kernel read that maximum from `shared_sum[0]` and
immediately reused the same workgroup array for squared values. One invocation
could overwrite the maximum before other invocations had read it. The missing
barrier could make a whole normalized value row zero.

The repair adds `workgroupBarrier()` between those two phases, through the
canonical WGSL variant patch. It preserves the arithmetic, storage types,
selection rules and numerical thresholds. The shader digest and its source
conversion reference are regenerated. A stale registry reachability entry from
the preceding LFM2 activation repair is also regenerated; the runtime closure
receipt changes only for that registry's content hash.

## Evidence

- Before: two fresh Apple M3 Node runs with identical prompt IDs and greedy
  sampling generated different continuations. Layer-0 V projection inputs were
  bit-exact; value normalization differed at rows 40, 44, 45, 46 and 56, with
  maximum absolute difference 13.265625. Selected arrays are retained here.
- A standalone operator regression reproduced a zero output where its independent
  reference expected 1.3704264. It uses deterministic synthetic inputs, so the
  regression does not require downloading model weights.
- After: 48 operator dispatches per surface passed in Node/WebGPU and
  Chrome/Metal, covering widths 64, 128 and 256 in immediate and recorded modes.
  All repeated results were bit-exact; maximum absolute error against the
  independent F64 reference was below 0.0015.
- Three fresh whole-model processes generated the same 16 token IDs. Two
  additional two-token runs captured 715 stage arrays across prefill and the
  first decode step; every corresponding array was byte-identical.

These are scoped repeatability results. The whole-model runs use the existing
isolated diagnostic overrides documented in `diagnostic-scope.json` and the
prior [request](../gemma-postnorm/diagnostic-request.json).
They do not change production guards or hosted artifacts, establish source-logit
parity, qualify AMD, or promote a catalog entry. Capture changes resource
lifetimes; the separate continuation runs use no TSIR capture, while retaining
the original diagnostic/trace settings. Individual captured stages do not form
one simultaneous model snapshot.

## Reproduce the operator regression

With the existing Node WebGPU provider installed:

```sh
node tests/inference/rmsnorm-f16-small-race.test.js
node tests/integration/rmsnorm-f16-small-race-browser.js --physical
```

The first command explicitly reports a skip when a Node GPU is unavailable;
a skip is not acceptance evidence. The browser check requires Chrome and rejects
software or unidentified adapters. Both call the same regression helper.

The full model's local diagnostic manifest SHA-256, source revision, shader and
test hashes, run projections and attachment hashes are bound in
`receipt.json`. Complete local capture files are not shipped in this packet;
`boundary-replay-summary.json` retains their hashes, and the four selected
layer-0 arrays are included. Original complete command envelopes remain local;
the retained projections identify them by SHA-256.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: the operator and continuation observations above, focused
RMSNorm/dtype ownership contracts, kernel checks, package closure/digests, source
style and architecture checks.
Boundary effects: shader execution identity and its generated references;
production capability policy and model publication unchanged.
