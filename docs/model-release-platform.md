# Doppler model release platform

Doppler's standalone product is the release platform for local models in
JavaScript. The supported unit is an immutable Model Pack plus its qualified
application and target evidence—not a model that merely loads and not a runtime
benchmark in isolation.

The validated contract is
`tools/policies/model-release-platform.json`. It records the Forge/Pack/Runtime
boundary, model-neutral ModelIR obligations, complete Pack target, provider
neutrality, Pack-first API migration, recovery obligations, commercial offer,
promotion sequence, compatibility graph, custody boundary, and explicit gaps.
`npm run model-release:check` rejects drift between that contract and the goal
completion matrix.

## Standalone offer

The first paid offer is **Model Release Qualification**:

> Give Doppler one pinned model and one real application. Receive a signed Pack,
> declared browser and Node qualification, known exclusions, and supported
> upgrade, rollback, requalification, and revocation procedures.

Generation, embedding, and reranking are the first application classes. The
free SDK, verifier, schemas, Forge tooling, and public examples create adoption.
Paid surfaces are private Packs, qualification campaigns, supported matrices,
private registries, recurring upgrades, incident response, tuning, and ongoing
release maintenance.

Commercial promotion remains unestablished. It requires an external application
to accept a qualified Pack, later accept an upgrade through the same process,
and depend on Doppler's qualification or revocation decision. An internal Pack
proof cannot satisfy that gate.

## Honest implementation boundary

Pack v2, `openPack()`, the dual-hexagon split, source-proven ModelIR machinery,
qualified TargetPlan selection, semantic hashing, signing, artifact validation,
and physical execution are implemented in bounded lanes. The contract also
records three unfinished groups:

- Pack closure still needs complete source/license, workload/oracle, exclusion,
  supersession/migration, upgrade preservation, and portable snapshot identity.
- The OpenAI-compatible server and lower-level generation, embedding, and
  reranking surfaces have not converged on one Pack-authoritative path.
- No external team has completed and repeated the paid release qualification.

These gaps are rows and blockers in
`src/config/goal-completion-matrix.json`. A passing contract check proves they
are accurately wired; it does not mark them complete.

## Provider hedge

Runtime selects only prequalified TargetPlans and cannot invent a plan on the
user's device. Doe is an optional provider, never a dependency. Dawn, ONNX
Runtime, WebNN, vendor-native paths, and a CPU reference may be qualified when
they best satisfy the application contract. Provider availability alone never
authorizes fallback.

## Data and acquisition boundary

Customer-private models, inputs, outputs, and application facts remain isolated
without explicit authority. Shared learning is limited to sanitized failure
signatures, minimized synthetic reproductions, public evidence, and
customer-approved derived patterns.

External release dependence is the operating objective. Acquisition interest
from model hubs, browsers, developer platforms, OEMs, silicon vendors, or
local-AI companies is a possible consequence, not the promotion gate.
