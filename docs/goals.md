# Doppler Goals

Doppler is an AI-native model release foundry and evidence-backed WebGPU runtime
for JavaScript. The immediate win condition is a reproducible open execution
network delivered with Poolday. Free adoption counts; payment does not determine
technical completion.

Doppler owns model semantics, source lineage, signed immutable Packs, artifact
integrity, qualified TargetPlan selection, and execution. Poolday owns participation,
authorization, assignment, transport, comparison, requester acceptance, and
admission of reusable evidence. The runtime never invents a plan or silently
changes signed model behavior.

## Goal 1: Reproduce an open execution network

ESM-2 35M, `esm2-t12-35m-ur50d-f32-af32`, remains the first network demonstration.
The connected capability is acquire, verify, execute locally, optionally
redistribute authorized artifacts, and explicitly delegate complete jobs.
Each selected execution peer runs the complete model. Downloading a Pack does
not authorize redistribution or delegation.

The demonstration uses actual signed Pack bytes, actual weights, and physical
browser execution through public `openPack()` and `encodeSequence()`. A fixed
vector, injected model program, local-tab identity, or successful contract test
does not establish that proof. Freeze a public-sequence corpus, requested outputs,
resource limits, and numerical correctness oracle before qualification.

One retained episode must connect:

1. A fresh receiver reconstructs the exact Pack and complete artifact closure
   from multiple authorized peers on independent machines, with origin and
   alternate mirrors disabled.
2. Corruption is rejected and participant loss is recovered without weakening
   integrity or silently fetching the origin.
3. Local execution and explicitly delegated complete jobs produce useful,
   oracle-valid, assignment-bound results under the same frozen acceptance policy.
4. Poolday retains outputs, disagreements, failures, review decisions, source
   attribution, transfer and execution costs, and the Pack/assignment lineage.
5. Admitted history changes a later held-out assignment and improves a predeclared
   meaningful outcome against both no-history and competent frozen reliability
   schedulers, with uncertainty reported and correctness unchanged.
6. Independent requesters and operators voluntarily return. Free use qualifies;
   protocol keys alone do not establish distinct machines or operator control.

All comparison arms receive the same jobs, eligible capacity, oracle, resource
budget, and controlled starting conditions. History precedes held-out jobs and
policy tuning ends before evaluation. Charge all attempts, replication,
verification, retries, failed transfers, relay traffic, and review. Report
resource quantities before optional monetary conversion. Revoked or expired
evidence loses influence; runtime changes require scoped requalification;
duplicates cannot manufacture support; insufficient history stays uncertain.

Private inputs remain local by default. The initial delegated workload uses
explicitly public protein sequences; no new private-input sharing permission is
implied. Protocols remain forkable, evidence exportable, and redistribution and
delegation separately opt-in.

Doppler and Poolday remain usable independently. Electron, Qwen, Bun, existing
reference integrations, and provider qualification retain their own supported
boundaries and evidence. They are not simultaneous network launch gates.
Doe improvements and agent-proposed Pack improvements are separate later
experiments and neither blocks this initial network. No scientific or biological
claim follows from matching embeddings.

## Retained standalone and commercial work

Doppler Production Release and Doppler Release Operations remain commercial
hypotheses for TypeScript/Electron products on Windows and macOS. Existing
Pack-first reranking, release tooling, application qualification, revocation,
and customer-controlled rollback remain usable. The application retains
activation authority; neither Doppler nor Poolday self-promotes a release.

The Electron document-search reference uses
`qwen-3-reranker-0-6b-q4k-ehf16-af32`. Its fleet, adoption, paid release, repeat
upgrade, and design-partner evidence remain separately tracked, not erased or
promoted. See [the standalone release contract](model-release-platform.md).
Revenue and acquisition are commercial hypotheses outside technical acceptance.

Provider neutrality and the runtime ownership comparison remain requirements
for provider superiority claims. They do not require winning an unrelated
portfolio comparison before the exact ESM-2 network can be demonstrated.

### Support ladder and lifecycle

Support is earned in levels. Lower levels remain visible engineering state, but
only `product-supported` may power the strongest public support language.

| Level | Meaning |
| --- | --- |
| `contract-ready` | The artifact validates and loads under its declared contract. |
| `runtime-verified` | Declared operations execute correctly on a named surface. |
| `task-qualified` | Frozen held-out task gates pass for the declared use. |
| `performance-qualified` | Declared latency, throughput, and memory SLOs pass. |
| `product-supported` | Package, surface, reliability, maintenance, revocation, and requalification commitments pass. |

Qualification level is separate from lifecycle. Catalog and support state must
also distinguish `candidate`, `active`, `deprecated`, `quarantined`, `revoked`,
and `retired`. Both are scoped to named surfaces and hardware classes. A
product-supported Node lane does not imply browser support, and a runtime pass
does not imply task quality.


## Completion matrix

`tools/policies/model-release-platform.json` and its strict schema encode the
network contract, commercial separation, and required evidence.
`npm run model-release:check` validates them against
`src/config/goal-completion-matrix.json`. The matrix has one
`technical-network` goal; retained `standalone-commercial` and `supporting`
goals do not add network launch gates.

`npm run product:readiness:report` reports network acceptance and its action
queue separately from standalone and supporting work. Structural validity does
not mean execution, independent adoption, or measured history benefit. Missing
evidence remains blocked and non-claimable.

Every blocker names its priority, owner, status command, exact exit criteria,
and completion authority: repository, application, hardware, production authority,
or human promotion. A passing check validates the contract, not the external
facts it requires. Existing support and claim checks remain enforced within
their declared scopes.

## Goal 2: Own the model artifact and runtime contract

Product contract:

- Users can load verified hosted models by registry ID or explicit model URL.
- The model support matrix states which model families, surfaces, and receipts
  are verified.
- Local caches are implementation details; artifact identity is still manifest
  and catalog owned.

Technical contract:

- RDRR manifests own model parameters, shard identity, tokenizer metadata,
  quantization, session policy, execution graph, kernel references, dtype policy,
  and artifact identity.
- Conversion-owned storage facts require conversion or manifest migration to
  change. Runtime config may not rewrite them.
- Runtime config may overlay only runtime-owned execution policy such as session,
  loading, diagnostics, and explicit kernel-path/session behavior.
- Defaults must be represented in schema, manifest, profile, config, or rule
  assets. Runtime code must not invent hidden behavior.

### Resolution identities

Every execution exposes three separate identities:

- `logicalModelId`: the application-facing model request.
- `resolvedArtifactVariantId`: immutable weights, tokenizer, quantization,
  graph, shard set, and artifact metadata.
- `resolvedExecutionId`: provider, kernels, runtime profile, cache policy,
  precision transforms, capabilities, and other execution decisions.

The catalog `manifestVariantId` names an intended artifact lane; it is not a
resolved identity and must never occupy either SHA-256 field. Before execution,
a candidate may bind the manifest variant while leaving resolved identities
null. Qualification requires the runtime-observed manifest and execution
digests.

A receipt binds all three. Applications must be able to accept
policy-authorized resolution, restrict allowed artifact or execution variants,
pin exact variants, reject every alternative, and inspect the final resolution.
Artifact identity must not change merely because the same bytes execute through
a different provider or runtime policy.

### Runtime ownership test

Every product-supported workload compares:

1. Source-model or authoritative reference execution.
2. The strongest relevant instrumented incumbent execution.
3. Doppler execution when Doppler proposes to own that path.

The incumbent may be Transformers.js, ONNX Runtime Web, WebLLM, LiteRT, or
another provider appropriate to the artifact and workload. The control must not
be selected because it is convenient for Doppler to beat.

The provider decision records correctness class, unsupported operations, task
quality, usability, memory, end-to-end performance, diagnostic depth,
distribution cost, integration burden, provider risk, and evidence identity.
Its disposition is `incumbent`, `doppler`, or `dual`. Evidence must name
predeclared material-advantage thresholds and expire when relevant artifact,
provider, browser, driver, adapter, graph, or kernel identity changes.

Doppler-owned execution is justified only where it provides at least one
predeclared material advantage, such as correct support unavailable in the
incumbent, a qualified memory or end-to-end improvement, actionable diagnostic
depth below the incumbent API, required offline/artifact-control behavior, or a
faster verified correction path. Equivalent incumbent behavior should be used
or supported rather than reimplemented for ownership's sake.

`benchmarks/vendors/runtime-ownership-decisions.json` is the machine-readable
authority, validated by `npm run runtime:ownership:check`. A claimable decision
must bind the exact source, incumbent, Doppler artifact and execution, frozen
correctness class, predeclared hypothesis and threshold, all decision evidence,
qualification date, expiry, and one `incumbent`, `doppler`, or `dual`
disposition. All semantic receipts share an exact harness/environment identity,
and publishing the disposition requires a human promotion receipt bound to the
evidence and hypothesis sets. An empty decision registry is valid but
incomplete. See `docs/runtime-ownership.md`.

Source and incumbent execution identities are canonical SHA-256 digests of the
provider-neutral execution receipts they reference. Doppler execution identity
remains the native runtime-observed SHA-256 in its local provider receipt. The
decision checker recomputes or cross-checks those identities rather than
accepting operator-entered execution labels.
`npm run runtime:ownership:record` can assemble a reviewable evaluation while
preserving frozen hypotheses, but it cannot make the decision claimable.
All retained ownership evidence is canonical-digest-bound. Decision dimensions
and material-advantage results use semantic receipts tied to the exact source,
incumbent, and Doppler executions; the checker derives their outcomes from
observations and rejects arbitrary files or independently authored pass flags.

### Provider independence

Provider conformance covers standard browser WebGPU, the selected Node provider,
Doe, and other eligible implementations. It exercises the same model contract,
declared operations, lifecycle behavior, evidence fields, and applicable
correctness class. Doe-specific optimizations are named lanes. Doe must never
become an undeclared requirement for Doppler.

Bun product support has its own portfolio-wide composition gate at
`tools/policies/bun-product-qualification.json`. Generation, embedding, and
reranking must bind current Bun/WebGPU identity, both public surfaces, lifecycle,
correctness and held-out quality, reliability, memory, cold/warm distributions,
incumbent eligibility, requalification, rollback, and revocation. A recorder may
assemble a non-claimable candidate; only digest-bound human promotion can move
all three workloads, the Bun support tier, and the release target together. See
`docs/bun-product-qualification.md`.

The machine-readable authority is
`tools/policies/provider-conformance.json`, validated by
`npm run provider:conformance:check`. The core product lanes are browser WebGPU
and the selected Node provider; Doe and other implementations are explicit
named lanes. Qualification begins at an exact workload tuple and requires
current operation, lifecycle, resolution-identity, environment, correctness,
and provider-receipt evidence. All evidence is canonical-digest-bound; semantic
receipts bind the exact tuple and the checker derives outcomes from observations
instead of trusting copied summaries. Provider promotion requires a human
receipt for the evidence set, while suite promotion separately binds the paired
provider set. Contract fixtures and Program Bundle parity do not constitute
provider qualification. `npm run provider:conformance:record` can assemble a
non-claimable candidate, but cannot promote or replace promoted state. See
`docs/provider-conformance.md`.

## Goal 3: Make correctness and performance evidence-backed

Product contract:

- Doppler says "this model works" only from verified model/support evidence.
- Doppler says "this lane is faster/slower" only from comparable benchmark
  artifacts with clean correctness and explicit disclosure.
- Candidate, diagnostic, experimental, and internal-only lanes remain visible but
  are not promoted as tier 1 product claims.

Technical contract:

- Benchmarks use shared workload contracts, command parity, normalized compare
  JSON, release matrices, and traceable artifact paths.
- Kernel paths, dtype policy, batching, readback cadence, and runtime profiles
  are explicit and receipt-visible.
- Claimable comparisons require prompt/sampling/cache/load alignment, non-zero
  decode, clean correctness, real artifact source identity, and one traceable
  artifact per claim.
- Debug and investigation profiles are useful evidence, but not public speed
  claims unless promoted by the benchmark policy.

Evidence must distinguish four correctness classes instead of treating exact
token identity as universal:

- bitwise or exact-token equivalence;
- tolerance-bounded numerical equivalence;
- semantic equivalence;
- held-out task-metric qualification.

The change class declares the required evidence before execution. A lane cannot
select a weaker correctness class after observing its result. Product SLOs and
competitive wins remain separate: a useful supported lane need not beat every
incumbent, while a fast lane cannot compensate for failed reliability or
quality.

### Scoped evidence and controlled generalization

Evidence begins at an exact model, artifact, provider, device, phase, shape,
and workload tuple. Wider claims require new aggregate evidence over a declared
conformance suite:

```text
exact tuple
  -> neighboring shapes
  -> hardware cohort
  -> model cohort
  -> explicitly declared support scope
```

Individual receipts remain authoritative. No tool may rewrite one tuple result
into family-wide authority. Kernel, profile, and graph changes should reference
immutable artifact weights rather than create redundant model artifacts.

## Bounded recursive improvement

Doppler's recursive self-improvement loop is an evidence-bounded optimization
workflow, not unrestricted source or production self-modification. It operates
on one declared scope at a time:

```text
observe -> diagnose -> hypothesize -> materialize candidate
  -> verify -> measure -> reject or recommend
  -> human approval -> shadow -> canary -> promote
  -> monitor -> retain or revoke -> index the result
```

The loop consumes prioritized product gaps: an SLO breach, buyer-blocking
unsupported workload, provider-conformance failure, material memory or cost
problem, or predeclared competitive opportunity. It must not optimize whichever
internal metric is easiest to move.

Every campaign freezes before execution:

- one falsifiable causal hypothesis;
- the metric expected to change;
- a control metric expected not to change;
- an end-to-end acceptance metric;
- neighboring-workload regression limits;
- artifact, execution, provider, hardware, and workload scope;
- candidate, resource, and measurement budgets;
- stopping rule, lineage, owner, and revocation conditions.

The first numerical divergence is a diagnostic location, not automatically the
causal performance target. Correctness runs before performance credit.
Promotion and activation remain separate operations. Every terminal candidate,
including invalid and rejected candidates, becomes indexed evidence with
explicit conditions under which retry is meaningful.

Evidence acquisition also has one canonical owner per job. Product integration,
provider conformance, runtime ownership, Bun qualification, post-promotion
monitoring, and signed revocation authority each use their declared
non-promoting recorder and independent checker. The
`evidence:workflows:check` script rejects duplicate owners, undeclared npm
commands, and missing owner or adapter paths; a convenience wrapper cannot
become a second authority. Revocation-authority drills must share one
harness/environment identity, and activation requires a human promotion receipt
bound to the complete canonical evidence set.

### Change-class evidence

| Change class | Required evidence |
| --- | --- |
| Scheduling, allocation, or cache mechanics | Exact output where declared, lifecycle, memory, end-to-end performance, and neighboring guards |
| Numerical kernel | Operator oracle, semantic boundary, deterministic continuation where required, and performance |
| Precision or quantization | Numerical evidence, held-out task quality, memory, and performance |
| New model artifact | Source equivalence under the declared correctness class, tokenizer and graph identity, task qualification, and reliability |
| Adapter | Activation proof, held-out improvement, resource impact, unload, rollback, and revocation |
| Provider integration | Conformance, artifact behavior, lifecycle, policy-authorized alternatives, and fallback visibility |

### Training boundary

General training and distillation are not required mainline product capabilities.
Gamma and external trainers may produce candidate checkpoints and adapters.
Doppler owns the consumable artifact contract, exact lineage, runtime activation,
held-out qualification, memory and latency impact, unload and rollback behavior,
revocation, and product promotion. Native browser training remains experimental
until a named buyer requires it and evidence shows an advantage over external
training.

## Observability, safety, and revocation

Product observability covers four linked levels:

1. Product health: load and usable-output success, cancellation, cache recovery,
   device loss, OOM, and API compatibility.
2. Model quality: declared correctness class, task metrics, finiteness, collapse,
   repetition, and human-review status.
3. Runtime performance: cold and warm load, prefill, decode, latency tails,
   throughput, submissions, readbacks, allocations, RSS, VRAM, and transferred
   artifact bytes.
4. Evidence health: receipt freshness, identity completeness, baseline validity,
   revoked dependencies, surface coverage, and public-claim drift.

Every observation binds artifact, tokenizer, graph, kernel, provider, runtime,
browser, adapter, driver, hardware, workload, sampling, and observation-policy
identity. It states whether instrumentation modified execution. Public or shared
telemetry is opt-in, redacted, retention-bounded, and local-first; raw prompts or
outputs are not exported by default.

Artifacts, tokenizers, WGSL, remote URLs, local-server access, caches, adapters,
and peer delivery are untrusted inputs. Qualification and distribution policy
must enforce validation, resource limits, origin rules, hashes, signatures where
required, and least privilege. Promotion must carry rollback and revocation
conditions. Revocation propagates to catalogs, quickstart resolution, claim
matrices, generated reports, and provider decisions without restoring a known
vulnerable predecessor.

## Product measures

The north star is reproducible useful execution across independent machines,
with admitted evidence improving later assignments and voluntary repeat use.
Supporting measures are scoped product outcomes, not raw catalog size:

- install-to-first-verified-output success;
- supported workload completion and usable-output rates;
- qualified model/surface/hardware coverage for the deliberate portfolio;
- cold and warm first-response, load, prefill, and decode distributions;
- crash, OOM, device-loss, recovery, and peak-memory behavior;
- held-out task-quality retention against the source model;
- artifact reproducibility and evidence freshness;
- voluntary repeat network use, including free adoption;
- regression detection, revocation, and requalification completeness;
- accepted improvement per measurement budget, with negative learning retained.

Candidate survival percentage is diagnostic rather than a north star: a low rate
may represent productive exploration, while a high rate may indicate that only
safe candidates were attempted.

## Ownership map

| Area | Goal |
| --- | --- |
| Open ESM-2 execution network, authorized custody, explicit complete-job delegation | Goal 1 |
| Release CLI, Electron adapter, fleet action | Retained standalone work |
| Browser execution, JS orchestration, WGSL kernels, Pack execution | Goals 1 and 2 |
| RDRR, conversion, manifests, catalog, hosted model IDs | Goal 2 |
| Runtime profiles, schema defaults, rule maps, kernel refs | Goal 2 |
| Release receipts, model matrix, subsystem tiers | Goal 3 |
| Vendor compares, benchmark SVGs, claim matrices | Goal 3 |
| Runtime ownership decisions and provider conformance | Goals 2 and 3 |
| Recursive improvement, promotion, monitoring, and revocation | Goal 3 |

## Non-goal boundary

Doppler executes and qualifies declared artifact and runtime contracts.
Orchestrators, agents, product loops, and application policy decide what to ask,
when to ask it, and how to use the result. Doppler determines whether evidence
satisfies its declared support and claim policies; applications determine
whether that evidence is sufficient for their use case.

Model-family expansion, direct-source proof lanes, LoRA, training, distributed
inference, P2P transport, program bundles, diffusion, hotswap, and orchestration
are supporting or experimental work until their own product qualification is
promoted. They do not become additional mainline goals merely because code or
exports exist.

The durable objective is a local inference system that people can use, extend,
compare, and audit without confusing available code with supported behavior,
successful execution with correctness, or a scoped measurement with a general
claim.

## Machine reporting

`src/config/goal-completion-matrix.json` is the source of truth for top-level
goal status. The matrix is validated by `npm run goals:check` and can be
rendered as a stable product-status report with `npm run goals:report`.

`npm run product:readiness:report` combines the goal matrix, command-surface,
model-artifact, maintained-integration, provider-conformance,
runtime-ownership, Bun product-qualification, policy-schema, claim-evidence,
revocation, promotion-monitoring, and subsystem-support contracts into one
markdown status report. A malformed Bun qualification now invalidates the
combined report even while an honest 0/3 experimental portfolio remains a valid
but incomplete state. Use
`npm run product:readiness:report -- --json` when another tool needs the same
status as machine-readable JSON. Revocation status separates the bundled
package authority from signed-live mechanism availability, qualification-
contract validity, and operational authority qualification; it must not
collapse those distinct trust states.
`npm run product:readiness:check` keeps that projection in the default gate.

`tools/policies/product-integration-qualification.json` remains the source of
truth for the three internally controlled reference integrations. `npm run
product:integrations:check` validates Reploid generation, Dream embedding
retrieval, and Columbo reranking; current named owners; the five-level support ladder; exact
logical identity plus runtime-observed `sha256:` artifact and execution
identities; expiry; incumbent controls; and
the required reliability, memory, quality, upgrade, rollback, and revocation
evidence. Evidence paths alone are insufficient: each reference binds a
canonical JSON digest and the checker validates its class-specific semantics.
`npm run product:integrations:record` can materialize a reviewed candidate from
retained receipt paths, but it preserves the declared support level, forces a
candidate lifecycle, and never enables a claim. Promotion remains an explicit
authority decision.
An empty registry is valid but explicitly incomplete.

`tools/policies/provider-conformance.json` independently governs provider
qualification. `npm run provider:conformance:check` requires generation,
embedding, and reranking suites across browser WebGPU and the selected Node
provider, while keeping Doe an optional named lane. An empty suite registry is
valid but explicitly incomplete; only current exact-tuple receipts can satisfy
the gate.

`benchmarks/vendors/runtime-ownership-decisions.json` governs whether Doppler
should own a workload path at all. `npm run runtime:ownership:check` requires
current generation, embedding, and reranking decisions backed by authoritative
source, incumbent, and Doppler execution evidence plus predeclared material-
advantage thresholds. Performance evidence is necessary where claimed but is
not sufficient by itself.

`src/config/revocation-registry.json` is the bundled deny-only authority for
logical models, resolved models, source checkpoints, weight packs, manifest
variants, exact manifest hashes, adapter IDs, and adapter source or execution
digests. `npm run revocations:check` validates the registry, retained evidence,
quickstart filtering, model and adapter catalog lifecycle, and withdrawal from
claims, integrations, provider suites, and runtime-ownership decisions. Model
identity is enforced before device initialization and weight loading. Adapter
identity is enforced before adapter bytes load and again before activation. No
named replacement is selected automatically. The root API also implements an
opt-in signed P-256 live-update mechanism with bounded fetches, expiring offline
state, persistent sequence and epoch checks, recovery-key rotation, monotonic
deny records, and loaded-identity invalidation. That mechanism does not make the
wider product claim complete: Doppler ships no qualified production endpoint,
package-trusted production key custody, qualified durable store, or retained
rotation and compromise-recovery drill receipts.

`tools/policies/signed-revocation-authority-qualification.json` is the source
of truth for that production boundary. `npm run revocations:authority:check`
requires one current, active authority binding an exact HTTPS endpoint,
authority ID, disjoint online and recovery key IDs, browser and Node durable-
state store identities, current security ownership, expiry, and repo-retained
evidence for deployment, package trust, custody separation, durable-state
behavior, every declared failure drill, application fail-closed behavior, and
requalification. All ownership and operational evidence is class-specific,
canonical-digest-bound JSON; the checker derives pass state and cross-checks
deployment identity instead of accepting path existence. Separate owner and
operational-evidence age limits bound qualification expiry. Its current candidate
deliberately leaves production facts null and `claimAllowed: false`; passing
the structural check does not qualify the authority.
`npm run revocations:authority:record` derives a candidate policy from retained
receipt paths but always preserves candidate lifecycle and disables claims;
recording evidence is not production activation. Passing 3/3 is reference
mechanics evidence and must not be reported as external adoption or commercial
demand.

`tools/policies/runtime-promotion-monitoring.json` closes the control loop after
a human promotion. `npm run promotion:monitoring:check` binds each promoted
candidate to its accepted optimization receipt, exact execution scope,
predeclared primary and control metrics, neighboring workloads, observation
evidence, known-safe rollback target, and original revocation conditions. It
recomputes `monitoring`, `retain`, or `revoke` without mutating production; a
revoke outcome requires a matching active revocation record. The mechanism is
implemented, and observation, rollback, and decision evidence is canonical-
digest-bound rather than accepted by path existence alone. Exercised promotion
coverage remains empty and non-claimable.
Human activation and terminal retain or revoke decisions are separate semantic,
digest-bound receipts tied to the candidate hash and exact scope.
`npm run promotion:monitoring:record` derives a monitoring policy record from
those receipts and observations without applying any runtime mutation.

The report intentionally treats a row as complete only when it is claimable,
has evidence paths, and declares an npm smoke command. Non-claimable rows must
name blocker codes and keep `smokeCommand` set to `null` so partial status is
explicit instead of implied by prose.

## Component intent contract

The repository [CATSCAN charter](../CATSCAN.md) translates these durable goals
into recursive component authority and invariants. A change inherits every
`CATSCAN.md` from the repository root to its target directory; child charters
may narrow but not contradict or broaden their parents.

The [generated component index](component-index.md) is the navigation surface.
`tools/policies/catscan-policy.json` owns the required charter inventory,
fields, word ceiling, and index path. `npm run catscan:check` validates unique
IDs, parent chains, local contract and evidence links, required sections, and
index freshness. Semantic alignment remains a review responsibility rather
than a claim the structural validator can prove.

`AGENTS.md` owns discovery, precedence, boundary-change, and handoff behavior;
it does not duplicate component goals.

## Policy schema registry

`src/config/schema/policy-schema-registry.json` registers the contract policy
schemas that support product-readiness checks. `npm run policy:schemas:check`
verifies that each registered policy exists, advertises the expected `$schema`,
and points at a strict JSON Schema 2020-12 document.

## Subsystem support contract

`tools/policies/subsystem-support-contract.json` points at the support-tier
registry. `npm run support:subsystems:check` verifies declared subsystem IDs,
docs, entrypoints, package exports, package bins, tier labels, and claim
visibility rules.

This keeps public product surfaces tied to concrete files and prevents primary
claims from drifting onto experimental or internal-only lanes.

## Claim evidence contract

`tools/policies/claim-evidence-contract.json` binds the release-claim policy,
benchmark policy, local inference claim matrix, release matrix, package scripts,
and goal matrix into one auditable evidence stack.

`npm run claims:evidence:check` verifies that release claims carry evidence and
performance report pointers, benchmark timing fields are represented in the
local claim matrix, release matrix source hashes are present, and the
correctness/performance goal cites the required evidence files.

## Model artifact contract

`models/catalog.json` is the canonical model contract. `src/config/quickstart-registry.json`
is a public quickstart mirror, not an independent source of truth.

`tools/policies/model-artifact-contract.json` defines the registry promotion
criteria: RDRR artifact, complete artifact metadata, manifest-owned runtime
promotion, no loose weight references, Hugging Face availability, active runtime
status, verified test status, pass result, and execution contract evidence.
`npm run artifact:contract:check` enforces that every exposed quickstart model
matches catalog fields exactly and that every catalog model satisfying the
promotion rule is exposed in the registry.

## Command surface contract

`tools/policies/command-surface-contract.json` records which canonical tooling
commands are browser-capable and which are Node-only. `npm run
commands:surface:check` checks that the policy matches the command API, CLI
usage text, and browser fail-closed guard.

This keeps the local WebGPU product-surface goal tied to the command contract:
browser-supported commands must stay available on browser and Node, while
Node-only commands must remain explicit failures on browser rather than silent
fallbacks.
