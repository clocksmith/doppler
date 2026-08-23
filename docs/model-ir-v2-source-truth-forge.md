# Heterogeneous ModelIR v2 and Source-Truth Forge

This document defines the semantic and provenance contract for bringing an
unfamiliar checkpoint into Doppler. It extends the immutable Pack architecture
in [ADR-0001](adr/0001-dual-hexagon-pack-spine.md); it does not create a second
runtime architecture.

## Objective

Forge must derive a faithful, compositional representation from pinned
checkpoint truth. Runtime must consume only a signed implementation contract.
Neither side may use a model name as a substitute for semantics.

The governing plane boundary is:

- JSON declares semantic truth, implementation policy, qualification scope,
  and permitted session values.
- JavaScript validates, binds, allocates, schedules, dispatches, and observes.
- WGSL performs only the declared tensor computation over explicit bindings.

Every reachable shader is a content-addressed member of one TargetPlan kernel
closure. A shader-byte or entry-point change therefore changes the TargetPlan
and Pack identity.

## ModelIR v2 shape

ModelIR v2 is a component and block graph rather than a flat decoder
configuration:

```text
ModelIR
├── source identity and provenance
├── components
│   ├── text decoder
│   ├── perception encoder
│   ├── connector
│   └── speculative drafter
├── block classes
│   ├── full attention
│   ├── local attention
│   ├── linear or recurrent attention
│   └── dense or MoE FFN
├── block schedules
├── state spaces
│   ├── KV state
│   ├── recurrent state
│   └── convolutional state
├── tensor-role bindings
├── entry points
└── output heads
```

Each block class owns its geometry, normalization, positional semantics,
persistent state, tensor roles, and legal phases. A block schedule references
classes; it does not copy one global attention contract across heterogeneous
layers.

The implementation is validated by `src/config/model-ir-v2.js` and
`src/config/schema/model-ir-v2.schema.json`. ModelIR v1 remains valid only for
its homogeneous scope.

## Provenance law

Every semantic fact records:

- an immutable source artifact;
- a JSON pointer, source pointer, or tensor-header observation;
- an attributable author;
- a confidence class;
- deterministic validation status; and
- a disposition.

Only direct or deterministically derived facts whose validation passed and
whose disposition is `accepted` may enter a signed ModelIR. Family inference,
ambiguity, unsupported structure, and rejected proposals remain evidence, but
they cannot be promoted as semantic truth.

`src/converter/source-truth-forge.js` applies this rule during intake.
`src/converter/lineage-lowering-forge.js` applies the same rule when a proven
lineage template is reused. Template reuse is permitted only after explicit
compatibility assertions pass against current source facts.

The source receipt retains the upstream semantic model identity. A lineage
receipt derives a Pack-bound ModelIR copy whose `modelId` identifies the
materialized artifact while `sourceIdentity` and every semantic fact remain
unchanged. Forge packages that receipt as `source-truth-evidence`, binds its
artifact ID into the Pack program, and verifies its bytes with the rest of the
artifact closure.

## Complete source topology and partial Pack scope

Forge preserves everything the source checkpoint contains. Product support is
represented separately:

1. `components` and `blockSchedules` describe complete source topology.
2. `entryPoints[].status` states which paths have a valid lowering.
3. TargetPlan qualification states the exact executable capability and target
   envelope.
4. Pack support scope publishes only those qualified entry points.

A text-only Pack may therefore represent perception and drafter components
while leaving their entry points unlowered. It may claim `text.generate`; it
may not claim complete multimodal or speculative support.

Forge promotes `qualifiedEntryPoints` only after a Program Bundle carries a
passed, exact source-token receipt on an explicit physical WebGPU surface. The
promotion requires exactly one lowered generation entry point, so evidence for
one path cannot silently qualify another.

## Lowering and promotion

Forge lowers semantics through reusable block capabilities. A lowering owns
phase programs and supported state kinds; it cannot be selected because a
checkpoint has a familiar name. Candidate proposals must be attributable.
Invalid candidates and losing valid candidates remain in the search receipt.

Promotion requires:

- a valid ModelIR v2 hash;
- a closed semantic execution graph;
- content-addressed WGSL and artifact closure;
- explicit dtype, layout, fusion, state, and memory policy;
- deterministic boundary and token evidence;
- a named hardware/surface qualification envelope; and
- an observed initial execution identity v2 whose signed `programLoadPolicy`
  uses policy schema v2 to recreate the qualified runtime session, compute
  policy, and multi-token decode setting, then matches the TargetPlan before
  first prefill dispatch.

`tools/run-program-bundle-reference.js --expected-transcript <path>` is the
promotion gate from an upstream reference to a Program Bundle. It records the
raw prompt-token comparison, the complete generated-token comparison, and the
first mismatch. A mismatch report is retained, but no Program Bundle is
emitted.

The Pack compiler accepts these promoted inputs explicitly:

```bash
node tools/forge-model-pack.js \
  --program-bundle <program-bundle.json> \
  --model-ir-receipt <model-ir-receipt.json> \
  --initial-identity <physical-qualification-report.json> \
  --qualification-report <physical-qualification-report.json> \
  --out <model.pack.json>
```

For ModelIR v2, Forge refuses specialization unless identity v2 was observed
before dispatch. Program Bundle reachability includes both phase-dispatched
kernels and `execution.mechanismKernels`; recurrent and convolutional
mechanisms therefore cannot execute outside the signed WGSL closure.

The observed identity binds the resolved graph, semantic-block mechanism
kernels, dtype lane, fusion set, KV layout, memory policy, execution plan, and
canonical resolved runtime-session digest. A later transition is still an
error; initial equality and subsequent immutability are separate gates.

## Campaign order

Qwen3.8 is the lineage-acceleration campaign. It may reuse established Qwen
semantics only where current checkpoint evidence confirms them. Changed block
classes, recurrent state, geometry, and tensor roles must be independently
derived.

Glimmer is the generalization campaign. Its complete source intake must pass
through the same pipeline before execution work begins. Glimmer execution may
add reusable component, block, state, or lowering vocabulary; it may not add a
Glimmer-named Runtime branch.

The campaigns are sequential. Qwen3.8 stabilizes the representation and
lowering contracts. Glimmer then tests whether they describe architecture
rather than one lineage.

## Release evidence

Each campaign retains:

- source-intake and ModelIR receipts;
- generated, rejected, and accepted candidate counts;
- unresolved facts;
- human interventions;
- accepted code identity;
- reference and Doppler transcripts;
- first-divergence boundary captures when parity fails;
- qualification and signed-Pack identity; and
- elapsed publication-to-qualified-JavaScript measurement.

The north-star measures are publication-to-first-correct-signed-Pack elapsed
time and the number of human-authored semantic decisions required for the new
architecture. A generated candidate count without retained rejection evidence
does not measure Forge quality.

## Current evidence boundaries

The permanent Gemma Pack Runtime v0 golden slice is frozen by tag
`pack-runtime-v0-gemma3-270m-golden`. Qwen3.8 `text.generate` now has a
development-signed Pack on physical Node WebGPU: the AMD/RADV receipt records
128/128 exact greedy tokens, 824/824 verified artifacts, initial execution
identity equality before prefill, and an unchanged TargetPlan digest. The
Release-to-JavaScript receipt under `reports/model-ir-v2/` binds its candidate
population, accepted code, unresolved source-publication time, explicit
development signer, and evidence bytes. It does not establish browser, Apple,
production-signing, multimodal, speculative, application, or Doe support.

Glimmer intake under the same directory proves representation construction
only and must not be described as execution support. Its deterministic
lowerability audit now compares source semantics with the admitted generic
vocabulary and fails closed on missing local attention, weightless embedding
normalization, scaleless Q/K normalization and query scaling, sigmoid gate
placement, centered post-normalization, no-RoPE full-attention, and final-logit
contracts. Those are reusable capability gaps, not permission to add a
Glimmer-named Runtime branch. The receipt now pins reference implementation
source hashes and source spans, places the pre-softcap multiplier on the output
head, and has zero unresolved text operational facts. Neither entry point is
lowered or qualified.
