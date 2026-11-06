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

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`,
`doppler.tests`. Intent: preserved. Boundary effects: recipe execution closure,
source-aware metadata refresh and variant identity, diagnostic observations.
