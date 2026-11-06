# Flash-Next learned-artifact qualification

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

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`,
`doppler.tests`. Intent: preserved. Boundary effects: recipe execution closure,
source-aware metadata refresh and variant identity, diagnostic observations.
