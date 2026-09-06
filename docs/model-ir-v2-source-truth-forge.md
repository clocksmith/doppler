# Heterogeneous ModelIR v2 and Source-Truth Forge

This document defines the semantic and provenance contract for bringing an
unfamiliar checkpoint into Doppler. It extends the immutable Capsule architecture
in [ADR-0001](adr/0001-dual-hexagon-capsule-spine.md); it does not create a second
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
and Capsule identity.

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
receipt derives a Capsule-bound ModelIR copy whose `modelId` identifies the
materialized artifact while `sourceIdentity` and every semantic fact remain
unchanged. Forge packages that receipt as `source-truth-evidence`, binds its
artifact ID into the Capsule program, and verifies its bytes with the rest of the
artifact closure.

## Complete source topology and partial Capsule scope

Forge preserves everything the source checkpoint contains. Product support is
represented separately:

1. `components` and `blockSchedules` describe complete source topology.
2. `entryPoints[].status` states which paths have a valid lowering.
3. TargetPlan qualification states the exact executable capability and target
   envelope.
4. Capsule support scope publishes only those qualified entry points.

A text-only Capsule may therefore represent perception and drafter components
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

### Candidate evaluation and ownership

`src/converter/execution-candidate-forge.js` owns semantic candidate generation.
Its v2 search receipt retains every eligible candidate. `acceptedCandidate`
without evaluation is a compatibility preview ordered by proposal score, not
correctness qualification or measured optimization; `selectedCandidates` stays
empty in that mode.

`src/converter/forge-candidate-evaluation.js` owns replayable evaluation and
selection. The explicit contract follows
`src/config/schema/forge-candidate-evaluation.schema.json`. It freezes candidate
hashes, ModelIR, source/oracle reference, inputs, runtime/environment scope,
cache/load policy, warmup/timed counts, rotation seed, complete output checks,
metric units, directions, and nullable limits. Retain the bytes behind every
digest, including source and oracle implementation, hardware/driver/provider
record, and exact runtime package. A digest alone proves none of those facts.

Use `createForgeEvaluationSchedule()` to obtain the ordered attempts.
`runForgeCandidateEvaluation()` invokes a host-owned `runAttempt` adapter and
retains raw observations through an awaited `onObservation` callback. The adapter
owns execution, cache preparation, work counts, timing boundaries, cancellation,
and cleanup; it must report observed scope and candidate identity, not simply
copy requested identities. Failed attempts do not retry silently. Cancellation
leaves unexecuted attempts visibly missing. Evidence-storage failure stops the
runner. No observer may change the frozen contract or source reference.

`evaluateForgeCandidates()` replays retained observations without a GPU.
Incorrect outputs, incomplete arrays, scope drift, missing attempts, and breached
limits exclude a candidate. Warmup outputs are checked but never timed. Order
changes, duplicates, and unknown attempts fail the evaluation. No successful
subset can replace the frozen population.

Selection is a conservative observed-range Pareto filter, per input case and
metric: one candidate dominates another only when its worst observed value is
no worse than the other's best everywhere and strictly better somewhere.
Overlapping ranges, ties, and latency/memory trade-offs retain alternatives.
These ranges are not confidence intervals or proof of general superiority.
Every receipt sets `claimAllowed` and `promotionAllowed` to false. Existing
benchmark statistical controls, physical qualification, and human promotion
remain separate; this filter does not replace them.

Pass the replay inputs as `evaluation` to semantic search or as
`candidateEvaluation` to `runForgePipeline()`. The latter binds exactly the
specialized TargetPlan hashes, preserves the selection receipt, and refuses
signing if none survive. Existing execution-identity and qualification gates
still run. Omitting evaluation keeps the existing closed-source-plan build
without granting optimization credit.

The existing file-based Forge command accepts `--candidate-evaluation <path>`
or `candidateEvaluationPath` in its JSON config. That JSON contains `contract`,
`reference`, and ordered `observations`. The command returns the replayed
`searchReceipt` and the input-file identity with its ordinary build receipt.
Keep the input and receipt together; neither is automatic public promotion.

Repository/host I/O remains in `src/tooling/model-capsule-forge.js`; compilation,
semantic decisions, and this selection algorithm remain under `src/converter/`.
The architecture gate prevents the Forge algorithm from importing model-host
composition or Capsule execution. This split is intentional, not two Forge engines.

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

The Capsule compiler accepts these promoted inputs explicitly:

```bash
node tools/forge-model-capsule.js \
  --program-bundle <program-bundle.json> \
  --model-ir-receipt <model-ir-receipt.json> \
  --initial-identity <physical-qualification-report.json> \
  --qualification-report <physical-qualification-report.json> \
  --release-manifest <capsule-release-v1.json> \
  --out <model.capsule.json>
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
- qualification and signed-Capsule identity; and
- elapsed publication-to-qualified-JavaScript measurement.

The north-star measures are publication-to-first-correct-signed-Capsule elapsed
time and the number of human-authored semantic decisions required for the new
architecture. A generated candidate count without retained rejection evidence
does not measure Forge quality.

## Current evidence boundaries

The permanent Gemma Capsule Runtime v0 golden slice is frozen by tag
`capsule-runtime-v0-gemma3-270m-golden`. Qwen3.8 `text.generate` now has a
development-signed Capsule on physical Node WebGPU: the AMD/RADV receipt records
128/128 exact greedy tokens, 824/824 verified artifacts, initial execution
identity equality before prefill, and an unchanged TargetPlan digest. The
Release-to-JavaScript receipt under `reports/model-ir-v2/` binds its candidate
population, accepted code, unresolved source-publication time, explicit
development signer, and evidence bytes. It does not establish browser, Apple,
production-signing, multimodal, speculative, application, or Doe support.

Glimmer's deterministic lowerability audit now admits `text.generate`: the
local- and full-attention block classes resolve to generic v2 lowerings, all
required state kinds are implemented, and the semantic manifest lowering
receipt binds the source facts, session policy, execution graph, and kernel
digests without a Glimmer-named Runtime branch. The source ModelIR still records
the original entry point as `unlowered`; the Capsule-bound ModelIR copy and
lowering receipt are the separate evidence that an execution candidate exists.

Physical Node WebGPU candidate reports now exercise that lowering on AMD/RADV.
They are investigation evidence, not qualification: no retained candidate
passes source-token parity. The BF16-storage candidate restores exact source
tokens through generation index 6 and first diverges at index 7; the committed
boundary comparisons remain diagnostic and `promotionEligible=false`.
`vision.encode` remains unlowered, and there is no qualified or signed Glimmer
Capsule, browser evidence, application evidence, multimodal evidence, speculative
evidence, or Doe evidence.
