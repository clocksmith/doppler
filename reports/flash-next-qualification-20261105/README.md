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
recorded separately. Full learned-model execution is still pending.

`capture-harness-failure.json` retains an instrumentation error from an
intermediate attempt: a custom emitter did not implement `getTimeline`. The
probe now uses the existing public `diagnostics` option and its real emitter.
That failed attempt is not numerical model evidence.

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`,
`doppler.tests`. Intent: preserved. Boundary effects: recipe execution closure,
source-aware metadata refresh and variant identity, diagnostic observations.
