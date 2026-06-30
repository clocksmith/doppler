# Doppler Goals

This file is the compact product and technical contract for Doppler's mainline
work. Generated matrices and receipts remain the source of truth for current
support and claims; this file states what the repo is optimizing toward.

## Goal 1: Make local WebGPU inference a real product surface

Product contract:

- Users can run verified local inference through the hosted browser demo,
  `npx doppler-gpu`, the root `doppler` API, CLI surfaces, Node/Bun execution,
  and the OpenAI-compatible localhost server.
- Tier 1 behavior is the verified text-inference path behind those surfaces.
- Advanced exports may exist, but support tier is defined by
  `docs/subsystem-support-matrix.md`, not by export shape alone.

Technical contract:

- One JS/WGSL WebGPU inference engine serves browser, Node, Bun, CLI, and server
  surfaces.
- Command semantics stay aligned across browser and CLI runners.
- JavaScript orchestrates load, prefill, decode, KV cache, streaming, and
  readback from resolved config.
- WGSL kernels perform deterministic math only.
- Unsupported environment or runtime capabilities fail closed instead of
  silently falling back.

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

## Ownership map

| Area | Goal |
| --- | --- |
| Browser demo, CLI, server, root API, Node/Bun surfaces | Goal 1 |
| JS orchestration, WGSL kernels, KV cache, streaming | Goal 1 |
| RDRR, conversion, manifests, catalog, hosted model IDs | Goal 2 |
| Runtime profiles, schema defaults, rule maps, kernel refs | Goal 2 |
| Release receipts, model matrix, subsystem tiers | Goal 3 |
| Vendor compares, benchmark SVGs, claim matrices | Goal 3 |

## Non-goal boundary

Doppler is the inference engine and artifact/runtime contract. Orchestrators,
agents, product loops, and policy layers decide what to ask, when to ask it, and
how to use the result. Doppler executes explicit contracts; it does not infer
policy from application intent.
## Machine reporting

`src/config/goal-completion-matrix.json` is the source of truth for top-level
goal status. The matrix is validated by `npm run goals:check` and can be
rendered as a stable product-status report with `npm run goals:report`.

`npm run product:readiness:report` combines the goal matrix, command-surface
contract, and model-artifact contract into one markdown status report. Use
`npm run product:readiness:report -- --json` when another tool needs the same
status as machine-readable JSON.

The report intentionally treats a row as complete only when it is claimable,
has evidence paths, and declares an npm smoke command. Non-claimable rows must
name blocker codes and keep `smokeCommand` set to `null` so partial status is
explicit instead of implied by prose.
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
