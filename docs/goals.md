# Doppler Goals

This file is the durable product and technical contract for Doppler's mainline
work. Generated matrices and receipts remain the source of truth for current
support and claims; this file states what the repo is optimizing toward and the
boundaries that must constrain that work.

Doppler exists to make local model execution usable, explicit, and auditable.
It is an evidence-backed WebGPU runtime for JavaScript applications, built for a
deliberately supported set of model workloads rather than maximum catalog size.
Application developers should be able to obtain a valid local result through a
small public surface. Runtime engineers should be able to establish exactly
which artifact, tokenizer, execution graph, provider, kernel, precision, cache,
and runtime policy produced that result.

The initial buyer is a JavaScript team shipping private, offline, or local
inference that cannot continuously maintain model conversion, browser
compatibility, hardware qualification, and model requalification alone. The
coherent product is a runtime SDK, qualified artifacts, compatibility evidence,
signed promotion and revocation state, and maintained integration support.

Runtime ownership must be proven rather than presumed. Artifact governance,
evaluation, application APIs, receipts, and requalification can also be built
around an incumbent runtime. Doppler should own execution for a workload only
when scoped evidence demonstrates durable value that an instrumented incumbent
does not provide. Otherwise Doppler should qualify, wrap, or interoperate with
the strongest eligible provider while applying the same identity and evidence
standards.

## Goal 1: Make local WebGPU inference a real product surface

Product contract:

- Users can run verified local inference through the hosted browser demo,
  `npx doppler-gpu`, the root `doppler` API, CLI surfaces, supported JavaScript
  hosts, and the OpenAI-compatible localhost server.
- Tier 1 behavior is the verified text-inference path behind those surfaces.
- Advanced exports may exist, but support tier is defined by
  `docs/subsystem-support-matrix.md`, not by export shape alone.
- Bun remains experimental until its own product-support evidence is promoted;
  its existence must not imply the same support as browser or Node.
- The initial portfolio should provide complete generation, embedding, and
  reranking stories for a small set of high-value Qwen and Gemma models.
  Biological encoders and specialist models should enter the product portfolio
  when a declared application requires them and their own qualification passes.

Technical contract:

- One coherent JS/WGSL execution contract serves supported browser, Node, CLI,
  and server surfaces. Experimental hosts remain explicitly classified.
- Command semantics stay aligned where a command is supported on multiple
  surfaces; unsupported commands fail explicitly.
- JavaScript orchestrates load, prefill, decode, KV cache, streaming, and
  readback from resolved config.
- WGSL kernels perform deterministic math only.
- Unsupported environment or runtime capabilities never trigger an undeclared
  fallback. An alternate lane is legal only when a policy explicitly authorizes
  it, the caller may reject it, and the resolution is visible in the receipt.

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

### Decisive product gate

The first decisive product gate is three distinct maintained application
integrations spanning generation, embedding retrieval, and reranking. Each
integration must use a product-supported lane, exercise upgrade or
requalification, preserve failure and rollback evidence, and have a named owner
who confirms that the integration remains active. Demos or multiple endpoints
inside one unmaintained example do not satisfy this gate.

Each integration must report install-to-first-verified-output behavior, exact
logical/artifact/execution identity, source-model task-quality retention, crash
and device-loss behavior, OOM and peak-memory evidence, cold and warm response
distributions, current browser/hardware qualification, and an incumbent-runtime
control.

## Completion matrix

The current completion state is encoded in
`src/config/goal-completion-matrix.json` and checked by `npm run goals:check`.
Rows are claimable only when `claimAllowed` is true, evidence paths exist, and
blockers are empty. Partial or experimental rows must name blocker codes, so
README claims cannot outrun support matrices, release receipts, or package
surface truth.

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
disposition. An empty decision registry is valid but incomplete. See
`docs/runtime-ownership.md`.

### Provider independence

Provider conformance covers standard browser WebGPU, the selected Node provider,
Doe, and other eligible implementations. It exercises the same model contract,
declared operations, lifecycle behavior, evidence fields, and applicable
correctness class. Doe-specific optimizations are named lanes. Doe must never
become an undeclared requirement for Doppler.

The machine-readable authority is
`tools/policies/provider-conformance.json`, validated by
`npm run provider:conformance:check`. The core product lanes are browser WebGPU
and the selected Node provider; Doe and other implementations are explicit
named lanes. Qualification begins at an exact workload tuple and requires
current operation, lifecycle, resolution-identity, environment, correctness,
and provider-receipt evidence. Contract fixtures and Program Bundle parity do
not constitute provider qualification. See `docs/provider-conformance.md`.

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

North-star measures are scoped product outcomes, not raw catalog size:

- install-to-first-verified-output success;
- supported workload completion and usable-output rates;
- qualified model/surface/hardware coverage for the deliberate portfolio;
- cold and warm first-response, load, prefill, and decode distributions;
- crash, OOM, device-loss, recovery, and peak-memory behavior;
- held-out task-quality retention against the source model;
- artifact reproducibility and evidence freshness;
- active maintained application integrations;
- regression detection, revocation, and requalification completeness;
- accepted improvement per measurement budget, with negative learning retained.

Candidate survival percentage is diagnostic rather than a north star: a low rate
may represent productive exploration, while a high rate may indicate that only
safe candidates were attempted.

## Ownership map

| Area | Goal |
| --- | --- |
| Browser demo, CLI, server, root API, Node/Bun surfaces | Goal 1 |
| JS orchestration, WGSL kernels, KV cache, streaming | Goal 1 |
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
runtime-ownership, policy-schema, claim-evidence, and subsystem-support
contracts into one markdown status report. Use
`npm run product:readiness:report -- --json` when another tool needs the same
status as machine-readable JSON.

`tools/policies/product-integration-qualification.json` is the source of truth
for the decisive maintained-application gate. `npm run product:integrations:check`
validates distinct active applications across generation, embedding retrieval,
and reranking; current named owners; the five-level support ladder; exact
logical, artifact, and execution identities; expiry; incumbent controls; and
the required reliability, memory, quality, upgrade, rollback, and revocation
evidence. An empty registry is valid but explicitly incomplete.

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
variants, and exact manifest hashes. `npm run revocations:check` validates the
registry, retained evidence, quickstart filtering, catalog lifecycle, and
withdrawal from claims, integrations, provider suites, and runtime-ownership
decisions. The runtime enforces matching records before device initialization
and weight loading and never auto-selects a named replacement. The current
authority is package-bundled, not live or cryptographically signed; that wider
claim remains blocked.

`tools/policies/runtime-promotion-monitoring.json` closes the control loop after
a human promotion. `npm run promotion:monitoring:check` binds each promoted
candidate to its accepted optimization receipt, exact execution scope,
predeclared primary and control metrics, neighboring workloads, observation
evidence, known-safe rollback target, and original revocation conditions. It
recomputes `monitoring`, `retain`, or `revoke` without mutating production; a
revoke outcome requires a matching active revocation record. The mechanism is
implemented, but exercised promotion coverage remains empty and non-claimable.

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

`models/catalog.json` is the canonical model contract. `src/client/doppler-registry.json`
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
