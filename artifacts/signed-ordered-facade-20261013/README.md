# Signed ordered resident facade

The public Capsule opener previously required legacy group methods and descriptor v1 for every resident. An admitted ordered allocation uses plan v3, descriptor v2 and executeStage, so opening failed before it could expose execution. The six initial controls in before.txt all failed; the final suite adds plan-mutation and in-flight cancellation controls.

The opener now selects required methods and descriptor schema from the validated allocation, checks ordered stageCount against the exact plan, and routes executeStage through the existing assignment and before/after live execution-identity checks. Ordered allocations reject legacy group methods; legacy allocations reject executeStage. Recovery, canonical admission, acquisition and cleanup owners are unchanged.

Evidence: tests.txt exercises actual signed Capsule verification with the repository test authority, synthetic qualification records and a synthetic program. Eight new cases cover all stages of 2/3/7-stage plans, descriptors, missing executor, acquisition cancellation, cleanup errors, input snapshots, execution identity mutation, invalid plan and in-flight cancellation. Existing canonical/legacy, paired opening, selective piece acquisition and factory cleanup suites also pass. This is CPU source contract evidence, not physical model qualification or a release grant.

Checks: source architecture, style, source types and CATSCAN. An initial command used the nonexistent npm script types:check; the actual typecheck:source command is used in the receipt. No archive, pin, reference or GPU execution changed.

Component: doppler.runtime-source.client. Intent: preserved. Boundary effects: versioned resident public facade through existing signed allocation and execution owners.
