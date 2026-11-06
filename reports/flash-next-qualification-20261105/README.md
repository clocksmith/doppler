# Flash-Next learned-artifact qualification

Current checkpoint: the QSA dispatch/query-view repair is pushed in `50135b70`.
The installed Node/AMD candidate passes the named public API and six short-input
lifecycle checks. Normal long-context numerical qualification still fails the
unchanged policy. Full-model availability and performance promotion remain open.

The initial Node physical load at runtime 603ef2dc was rejected before weight
allocation: the sigmoid source contract included SiLU gated-delta mechanisms.
The recipe now declares the existing sigmoid kernels and their canonical entry
digests. The integration regression exercises the same norm-gate closure guard.

Manifest refresh accepts an explicit source configuration for text-only artifacts
and rejects source/artifact geometry disagreement. It also recomputes the derived
manifest variant identity when the execution graph changes, while retaining the
unchanged weight capsule. No converted weight needs rebuilding for this fix.

The learned diagnostic optionally captures short-input layer outputs against
the unchanged upstream reference. Those observations are diagnostic, with no
post-hoc numerical tolerance or performance claim. Qualification remains pending
until provisioned physical runs complete. See retained failure and focused tests.

The provisioned retry then identified two materialization boundaries: F16 small
vectors were not converted for F32-only mechanism kernels, and declared projection
consumers were ignored when artifact classification was `other`. Both are fixed
without changing stored weights. Exact vector values and tracked ownership pass
on physical Metal and AMD; loader failure/ownership regressions also pass.

Routing exposed the next boundary: 512 experts exceeded the selected 256-expert
kernel. The recipe now declares a registered strided scalar 512-expert program.
Its immediate/recorded paths match independent CPU selections and probabilities
at 63/256/257/511/512 experts, normalized and unnormalized, including ties, on
physical Metal and AMD. Chrome/Metal checks the 512-expert path separately.
The existing 256-expert vector-component writes also produced wrong selections
on Metal; independent scalar workgroup storage repairs that defect. Its failing
and passing physical receipts are retained. No performance gain is asserted.

The operation catalog now recognizes the already-implemented `moe_topk` executor;
manifest admission still rejects unknown operations and unregistered constants.
Refresh validates metadata before expensive Merkle reconstruction. The focused
run reports 25 passing checks, with kernel/type/architecture/import/package checks
recorded separately. These component checks alone do not qualify the learned model.

`capture-harness-failure.json` retains an instrumentation error from an
intermediate attempt: a custom emitter did not implement `getTimeline`. The
probe now uses the existing public `diagnostics` option and its real emitter.
That failed attempt is not numerical model evidence.

## Bounded learned-weight execution

The unfused shared-expert addition returned a flattened tensor despite its
declared `[tokens, hidden]` result. The repair preserves the explicit geometry;
physical Metal and AMD checks cover that branch. Immediate decode also exposed
the sampler's padding mask to `onLogits`, unlike prefill and fused decode. A
behavioral regression reproduces the resulting `-Infinity` observation and
checks the repaired pre-suppression observation contract.

The fully resident run exhausted Jupiter's per-device GTT allocation and lost
the device. Raising the separate TTM page setting did not raise that device
limit. The new explicitly selected paging profile retains two complete Qwen
expert banks, settles submitted work before retirement, invalidates both alias
maps, and preserves cleanup failures. Recorded paging is rejected. The original
TTM setting was restored and verified; 27B remains active and 35B remains stopped.

Node/AMD runs now complete the learned 48-layer model under that original limit:

- 15 input tokens, three output tokens: `Paris. Water`, first ID 11751.
- 2,304 input tokens, two output tokens: `hidden_states`, first ID 7920.

Both first IDs match the pinned upstream references. All observed logits are
finite, both runs retain only layers 46 and 47, and neither reports a GPU or
cleanup error. The short run reports a buffer-pool peak of 13,811,447,808 bytes.
See [retained receipts](paged/receipts.json), compressed full observations and
logs in `paged/`. These are execution observations, not throughput measurements
or general model qualification.

Independent controls preserve the original source-weight reference and add
stored F16/Q4_K weights decoded by `gguf`, followed by the unchanged upstream
forward. A separately named control rounds post-RoPE keys and values through
F16 to match the declared runtime cache. Numerical comparisons use the existing
versioned boundary policy, without fitting a new tolerance to the output.
The F16-KV control reduces the largest short-input sampled layer discrepancy to
0.000137925, with zero quantized-policy violations across 184,320 values in all
48 layers. [Comparison](paged/short-kv-matched-comparison.json) and
[independent control identity](paged/independent-control-identity.json) retain
the policy and producer hashes. A deliberately corrupted coordinate is rejected.
This checks sampled activations, not every vocabulary logit. Full-logit,
installed-package and remaining physical lifecycle checks are recorded separately
from numerical agreement.

## Live QSA dispatch and the remaining long-input failure

Long-context numerical qualification remains blocked. The prepared QSA selection
was omitted when constructing both the live attention dispatch and its semantic
plan, so the model ran dense attention. The regression now exercises that real
handoff for prefill and decode. The repaired route also accepts the upstream
normalized query's `[tokens * heads, headDim]` view as `[tokens, heads * headDim]`,
after validating its declared geometry; it preserves the same buffer and bytes.

On physical Node/AMD, the first sparse layer's maximum sampled error falls from
0.00012014783 to 0.00000115484. All seven sampled layer-3 sparse selections match
the independent artifact/F16-KV reference. Instrumented and uninstrumented runs
produce the same complete prefill-logit hash. The default corrected run still
fails the unchanged full-vocabulary gate: maximum error 0.774933815, with 64,993
coordinates outside the existing policy. See [comparison](qsa/layer3-comparison.json)
and [compressed observations](qsa/receipts.json). Matching first tokens does not
clear that failure. Earlier operational receipts predate this routing repair.

The next diagnostic captures every router input and logit in the first eight
layers, rather than only sampled positions. Its independent CPU prefix matches
the retained control exactly at every retained sample. The first changed expert
set is layer 4, token 1,017: the F32 router reverses experts 218 and 260 across a
4.77e-6 source-logit margin. A physical replay at the original batch geometry
reproduces that reversal; the existing compensated F32 kernel instead matches
nearest-F32 Float64-reference results for every checked output. The explicitly
selected full-model compensated-router candidate still fails the unchanged
logit gate. It is a diagnostic candidate, not a production recipe change.

The source-only prefix attempt used the wrong producer and is rejected; only
the artifact/F16-KV control identity in `qsa/router-control-identity.json` is
used above. No comparison tolerance, stored weight or production precision
default changes in this repair.

## Repaired-runtime operational checks and routing isolation

The private installed package `c7dc6afdff57…` passes the public compatibility API
check: exact artifact/variant identities, execution receipts, three short-input
tokens and two long-input tokens, finite logit observations, and clean unload.
The short prefill and both decode rows pass the existing full-vocabulary policy
(maximum errors 0.000394, 0.000445 and 0.000655). The normal long prefill and
decode still fail (maximum errors 0.774934 and 1.133989). This archive is an
unpublished candidate, not an npm release or Reploid availability claim.

All six provisioned short-input lifecycle cases now pass with repaired QSA:
queued cancellation, active-decode rollback of committed KV/recurrent/PLE/QSA
state, reset/reuse, paused cancellation, unload of a paused consumer, and fresh
reload/reuse. No GPU or cleanup error is reported. The long input also executes
prefill and decode through the installed API; that does not extend the short
lifecycle cases into a long-context rollback qualification.

All 48 loaded router matrices (62,914,560 F32 elements) are byte-identical to
independent NumPy expansion of their declared F16 artifact tensors. Actual
source softmax/top-k and actual registered GPU softmax/top-k replay confirm the
first changed expert set at layer 4/token 1,017; this is not merely an inference
from sorting raw logits. Future prefix captures retain the upstream router's
actual output indices and weights directly.

The first reversed router row also passes the existing `source-f32-v1`
operator tolerance against a Float64 dot product of the exact same GPU input
and independently expanded F16 weights: maximum error 1.71733e-5, zero policy
violations. Experts 218/260 have an exact-input gap of +9.61687e-6, but the
GPU gap is -2.38419e-6. This establishes rounding sensitivity at that boundary;
it does not excuse the full-model failure or establish that every later
difference has the same cause. See [row assessment](qsa/router-rounding-assessment.json).

An explicitly counterfactual run replaces all router logits with independent
CPU fixtures. The remaining GPU stack then has maximum full-vocabulary error
0.00205517 with zero violations under the unchanged policy. The normal
qualification comparator explicitly rejects such controlled observations.
This isolates sensitivity to router outputs/choices; it does not qualify normal
generation, or prove that changing the acceptance criterion is appropriate.
The comparator also requires the GPU input hash to match the named frozen
reference fixture. CPU-only rejection checks cover counterfactual routing and
mismatched inputs; the ordinary retained
prefill/decode comparison reproduces its earlier result exactly. See
[comparator checks](qsa/comparator-contract.log).

The router-only compensated candidate fails the normal gate. A broader
full-matrix candidate ends on device loss coincident with an AMD graphics-ring
timeout and automatic reset; that run provides no numerical evidence and its
process PID was not retained. Subsequent probes retain PIDs for attribution.
Bounded 256/512-row control-projection probes match nearest-F32 references at
all checked coordinates. A normal candidate using 512-token prefill chunks and
compensated control projections (PLE projections unchanged, no CPU injection)
completes cleanly but still fails: maximum error 0.615987, 42,033 violations.
None of these candidates changes a production recipe or precision default.

The additional raw-distribution metrics are diagnostics, not a replacement gate:
the long prefill/decode top-20 overlap is 19/20 and reference-to-GPU KL is
8.61e-5/1.70e-5 nats on these particular prefixes. Remaining numerical work must
assess routing-sensitive F32 differences against the independent reference;
normal long-context qualification and performance promotion remain open.

See `qsa/public-consumer.json`, `qsa/public-decode-comparison.json`,
`qsa/repaired-lifecycle.json`, `qsa/router-weight-byte-comparison.json`,
`qsa/actual-source-gpu-topk-comparison.json`, and
`qsa/reference-router-control-comparison.json`. Raw model output and large
CPU fixtures remain on the external models volume, identified by these receipts.
The counterfactual and bounded-candidate observations, including their complete
runtime configurations, are retained as `qsa/reference-router-control-observation.json.gz`
and `qsa/bounded-compensated-controls-observation.json.gz`.

Focused development checks pass: the QSA dispatch, query-view, attention-plan
and cache-lifecycle files report nine passing checks; changed JavaScript syntax
and Python compilation checks pass; dependency freshness and source architecture
checks pass. The counterfactual helper's CPU-only preflight checks reject an
unmatched manifest, recorded execution and wrong geometry before dispatch, and
verify hook restoration. These checks do not add physical qualification scope.

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`,
`doppler.runtime-source.inference.pipelines.text`, `doppler.tests`.
Intent: preserved. Boundary effects: recipe execution closure, source-aware
metadata refresh and variant identity, repaired QSA dispatch/query views,
private package source increase of 955 bytes, and explicitly counterfactual
test observations. The failed broad candidate coincided with a driver reset;
no manual GPU reset, participant restart or persistent GPU-limit change was made.
