# Resident contract integration candidate

This batch implements Doppler's D1 public-type exports and D2 recovery declaration
from the Doppler–Reploid distributed-execution specification. Reploid's identity,
adapter, journal and scheduling changes remain owned by its separate agent.
No Reploid files were edited, and no physical inference was run here.

## Consumer handoff

Install the exact [candidate archive](doppler-gpu-0.6.27.tgz), whose digest is in
[the installed receipt](installed-receipt.json). This is a repository-hosted
integration candidate, not an npm publication or distributed numerical release.
It preserves model code, weights, shader selection, references and tolerance.

Import types from `doppler-gpu/partitions`, including:

- `ResidentPartitionDescriptor`, `ResidentPartitionIdentity`, `ResidentPartitionLimits`;
- `ResidentPartitionARequest`, `ResidentPartitionAResult`, `ResidentPartitionBRequest`, `ResidentPartitionBResult`;
- `ResidentPartitionTokenizationRequest`, `ResidentPartitionTokenizationResult`;
- `ResidentPartitionMetrics`, `PartitionTiming`, `ResidentPartitionSession`, `ResidentRecoveryCapabilities`.

`getDescriptor()` keeps `generationDigest` and `ready: boolean`; consumers validate
the accepted allocation before narrowing readiness. `getRecoveryCapabilities()`
returns an immutable `doppler.resident-recovery/v1` object with `inputReplay`,
`checkpointExport`, and `checkpointImport` all false. The verified Capsule wrapper
preserves this declaration. Missing methods, incompatible descriptor/recovery
versions, omitted flags and enabled recovery claims reject during opening.
Older residents without this method do not satisfy the updated interface.

Reploid should extend its existing installed-package signed-opening test against
this archive, then use the public types in its adapter. Its work may retain grants,
reservations, routing and transport envelopes as separate types. No second wire
protocol or checkpoint implementation is introduced.

## Acceptance and retained failures

[Source type checks](d2-types.log), [contract tests](d2-contract-tests-final.log),
and [isolated installed checks](d2-package-isolated.log) cover public declarations,
signed qualification rejection, malformed descriptors, missing methods, recovery
capabilities, device loss, closure and foreign continuation rejection. The installed
checks compile against the packed package rather than repository declaration paths.
Synthetic execution and physical inference remain separate claims.

The earlier type-export batch is commit `39b10b18`. Its first installed check was
placed inside the workspace, where parent dependency resolution invalidated the
absent-provider control. [Failure](d1-package.log) and [receipt](d1-failed-receipt.json)
are retained; the failed directory was moved to `/tmp/doppler-partition-contracts-d1-failed`.
The [external consumer run](d1-package-isolated.log) passes that control.

The initial [repository check](check-green.log) retains stale API inventory and
package-size budget failures. The inventory was regenerated. The package budget
now uses the exact [archive inventory](npm-pack.json), retaining its existing npm
compression allowance; no numerical acceptance threshold changed.
[API](api-docs-final.log) and [public-boundary](public-boundaries-final.log) checks
record successful targeted reruns. The original full command exits nonzero for
those two failures; every other stage, including all unit files, passes. It is
not rewritten as a successful original invocation. Runtime closure was regenerated without adding inference
ownership to the minimal root. The source and artifact sums bind this handoff.

Implementation is pushed in `39b10b18` (public types) and `4e85ce76` (recovery,
admission, candidate archive). The [admission cleanup regression](admission-cleanup.log)
also verifies that each rejected loaded program is closed. `SHA256SUMS` binds
the retained files; `source-SHA256SUMS` binds the reviewed implementation and
tests. The archive has been checked byte for byte against current shipped files.

Historical numerical failures and the frozen reference remain unchanged.
Both placements, repeated prompts, submitted cancellation, bounded memory,
resident reuse and recovery still require physical acceptance of one agreed
archive. Independent-unit embedding orchestration is Reploid-owned and requires
an explicitly separable application. Portable checkpoints remain deferred.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.client`, `doppler.repository-tooling`.
Intent: preserved. Acceptance evidence: the linked contract, type and package
checks. Boundary effects: public resident types and capability reporting;
no numerical execution, transport, assignment or artifact-storage changes.
