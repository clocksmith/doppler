# Resident partition execution

Doppler executes assigned transformer layers through the same loader and declared
GPU kernels used for unsplit inference. The public `doppler-gpu/partitions` source
entry exposes resident factories, layer-plan contracts, verified piece storage,
and device allocation observations. This guide describes implemented behavior;
implementation alone does not establish distributed release qualification.
The published npm 0.6.1 archive does not expose this entry.

Doppler owns numerical execution, resident weights, and attempt execution state.
Reploid owns discovery, grants, placement, transfer, readiness, conversations,
and retries; see the [consumer acceptance contract](https://github.com/clocksmith/reploid/blob/main/docs/chat-mvp.md#resident-integration-contract).
The [September 27 local diagnostic](../../reports/resident-partitions/20260927/README.md)
and [session checkpoint](../../reports/resident-partitions/20260927-session-checkpoint/README.md)
retain their historical source and incomplete acceptance boundaries. Their former
next-action lists are not current implementation status.

## Implemented public session boundary

`createResidentPartitionFactory({ openCapsule, capsuleOptions })` delegates to the
normal verified Capsule opener with explicit host trust. Opening supplies model
and executable identity, partition plan/hash/index, participant identity,
generation settings, allocation limits, and an abort signal. The returned
session exposes `getDescriptor()`, `tokenize()`, `executeGroup0()`,
`executeGroup1()`, `getRecoveryCapabilities()`, `closeAttempt()`, and `close()`.

Consumers import resident descriptors, identities, limits, step requests/results,
tokenization requests/results and metrics as types from `doppler-gpu/partitions`.
These are the computation contract; applications retain their own grant,
reservation and transport-envelope types. `getDescriptor()` includes the accepted
`generationDigest` and a live `ready: boolean`. Narrowing readiness to `true`
requires validating the descriptor against the accepted model, plan, index,
layer range and generation configuration. Package version, exported API,
resident readiness and signed model qualification are separate checks.

Each full attempt identity owns independent KV and recurrent state, continuation,
sequence position, generation context, and decoder. Calls validate identity,
step ordering, context bounds, activation shape/dtype, and generation settings.
Group 0 embeds token IDs and returns activation bytes; group 1 computes logits,
samples with the bound settings and authorized token context, and returns token,
decoded delta, completion reason, continuation, and observations. Final stopping
flushes the incremental decoder. A failed attempt is retired; restart it from
its prompt rather than inventing a missing recurrent prefix.

Paired intermediate execution has a distinct boundary identity. A version-two
layer plan declares `activationPrecision: 'two-f32'`, `activationComponents: 2`,
and `activationElementStrideBytes: 8`, with `activationDtype: 'f32'`. Its tensor
contracts and version-two activation frames declare `precision: 'two-f32'`,
`components: 2`, and `elementStrideBytes: 8`. Shape remains logical
`[batchSize, seqLen, hiddenSize]`; the contiguous payload stores each high F32
word immediately followed by its low F32 word. No JavaScript tensor conversion
occurs. The immutable resolved session must explicitly select matching
`intermediatePrecision`; the bound execution plan must independently declare
matching paired precision, component count, and stride. A requested session alone
cannot enable paired execution with scalar kernels. Signed plan digests, byte allowances, and program-load
identities therefore distinguish this path. Public logits remain F32.

Existing version-one F32/F16 plan hashes and frames are unchanged. A paired frame
cannot be downgraded to version one or accepted with missing/mismatched precision,
component count, stride, or byte length. Partition chunking preserves all words;
its input/output allowance counts eight bytes per logical paired element.
These contract and ownership checks do not qualify the new arithmetic, a model,
a signed artifact, or either physical distributed placement.

Closing an attempt settles pending work before releasing its state while retaining
resident weights. Session close settles attempts and closes the owned program.
Submitted GPU commands are not interrupted by cancellation; a cancelled operation
suppresses successful completion and retains resources until settlement.
These source mechanisms need exact-package lifecycle evidence on each claimed host.

`createManifestResidentPartitionFactory({ manifest, manifestIdentity,
runtimeConfig, createStorage })` is an explicit development lane. It requires a
pinned manifest, exact digest, explicit runtime configuration, and injected
verified artifact storage. It preserves numerical and allocation checks but does
not claim signed Capsule or partition qualification. See the
[public declarations](../../src/client/resident-partitions.d.ts) and
[session contract](../../src/inference/pipelines/text/resident-partition-contract.d.ts).

Resident Capsule opening requires a signed TargetPlan v2 qualification record
for each assigned group on the observed surface. The record's
`operation: "residentPartition"`, `partitionPlanHash`, `partitionIndex`,
`comparedSteps`, `transcriptHash`, and evidence artifact bind that exact split
to the signed plan. Whole-model `generate` evidence cannot authorize either
group. The Capsule root checks this record before constructing a program; the
model host supplies the pure manifest-bound allocation validator through a
port, keeping the root independent of the inference executor.

## Signed selective acquisition

An explicitly prepared Capsule may set `program.pieceIndexArtifactId` to an
artifact with role `piece-index`. Its bytes use the existing
`doppler.verified-pieces/v1` format and bind the exact signed manifest digest.
The Capsule signature covers this reference and artifact identity. The existing
`tools/build-verified-piece-index.js` producer checks complete source shard hashes
before producing piece hashes; a host cannot supply an unsigned replacement index.

Only resident opening with matching signed partition qualification enables this
path. The host supplies `artifactStore.readArtifactRange(artifact, offset, length,
options)` for complete requested pieces. Doppler snapshots and hashes each piece
before returning any range, checks coverage and signed shard identities, and
limits piece size by the host's acquisition policy. No unrelated shard-tail probe
or whole-shard acquisition is required. An absent index, and every ordinary
nonresident opening, preserve complete artifact verification.

`session.verification.artifactVerificationScope` is `resident-pieces` on this
path. `artifactReceipts` contains only completely hashed metadata artifacts;
`pieceVerification` identifies the signed index and authenticated piece ranges
actually acquired during preparation (`observation: 'resident-opening'`). This
frozen opening snapshot is not a live execution counter. Every later read still
authenticates its pieces. These receipts do not claim full weight-shard hashing.
Incomplete verification is never written to the whole-closure verification cache.
Storage closure aborts owned reads and settles their wrappers, including when a
host source ignores cancellation. It does not wait for an uncooperative source.

This is an additive schema field and artifact role, shared by Capsule v2/v3.
Older consumers reject new records rather than silently ignoring the integrity
mode. Existing records remain compatible. New signed records are not generated,
qualified, selected, or published automatically. The
[synthetic opening controls](../../tests/runtime/resident-piece-acquisition.test.js)
prove acquisition and rejection contracts; they do not establish physical model
qualification or release acceptance.

## Recovery capability contract

`getRecoveryCapabilities()` returns an immutable `ResidentRecoveryCapabilities`
through `doppler-gpu/partitions`:

```js
{
  schema: 'doppler.resident-recovery/v1',
  inputReplay: false,
  checkpointExport: false,
  checkpointImport: false,
}
```

These are capability facts, not configurable switches or grants. This initial
implementation supports none of the declared recovery operations. Capsule
opening rejects missing resident methods, unsupported descriptor/recovery
versions, missing capability fields and enabled recovery claims. Upgrading the
consumer requires the matching archive; an older resident without this method
does not satisfy the new interface. The resident descriptor retains its existing
schema, generation digest and live boolean readiness.

A continuation's nonce, step and position bind in-memory state in its original
attempt. Process loss cannot export that state; a lost device or closed session
rejects execution. A replacement resident rejects a foreign continuation even
when its position is rewritten to the initial step. Restart means a new attempt
from approved inputs, not appending to the prior partial output invisibly.

| Operation | Owner and current boundary |
| --- | --- |
| Artifact resume | Existing acquisition/storage path resumes verified model bytes. |
| Outcome replay | Reploid may redeliver committed job outputs under current authorization. |
| Restart | Reploid creates a new assignment/attempt; Doppler executes fresh state. |
| Input replay | Unsupported reconstruction from committed inputs/tokens. |
| Checkpoint restore | Unsupported import of serialized inference state. |

Reploid owns attempt history, recovery authorization and placement changes.
Transport interruption never establishes computation completion or resource
settlement. Portable checkpoints and expansion beyond the existing partitions
remain separate capabilities. Contract tests use synthetic execution and cannot
replace physical numerical or lifecycle qualification.

## Current numerical boundary

`createPipeline(manifest, { partition: { plan, index }, ...contexts })` uses the
existing partial weight loader and allocates only its assigned KV layers. Cache
layout is resolved against the complete model before allocation is narrowed.
Original layer indices remain authoritative for weights, attention, and RoPE.
Mixed geometry, shared KV and non-contiguous resolved cache layouts currently
fail closed for partition allocation.

`partition-execution.js` runs under the existing `runPipelineOperation` lease.
Group A embeds token IDs and returns owned activation bytes. Group B uploads
those bytes, executes its layers, and returns owned GPU logits. Existing
embedding, layer, normalization and logits kernels perform the computation.
No model names, CPU tensor fallback, whole-model fallback, or production tuning
switch chooses the split. Precision must match the resolved plan.

The caller owns the sequence position, cache and returned logits buffer. It must
release the logits and retire the attempt after execution failure: recording may
already have advanced cache metadata. This low-level API is not a security or
concurrency boundary and must not be passed directly to Reploid as a resident.
Legacy GPU execution stays serialized. The executor checks cancellation before
dispatch, between layers, after completion and after readback; submitted work
settles before its temporary resources are released.

Admission allows token-only causal incremental full, sliding, and linear attention.
Linear attention requires its recurrent state to exist and match the sequence
position before dispatch. MoE, shared KV, per-layer inputs, adapters, multimodal
inputs, cross-layer normalization fusion, and finiteness fallback transitions
are rejected. Token-only use of a model with additional modality capabilities is
not the same as qualifying its multimodal inputs.
The retained numerical evidence covers only the exact artifact and precision
specified in its report.

## Physical evidence and remaining acceptance

The [current recovery investigation](../../artifacts/recovery-20261009/README.md)
retains operator captures, independent calculations, exact archive identities,
and two-physical-GPU comparisons. Retained package comparisons still exceed the
frozen 0.001 tolerance; the owning report records exact candidates and failure
outcomes. Bounded operator agreement is diagnostic evidence, not package
promotion. Do not infer acceptance from selected-token agreement alone or from
an earlier candidate's outcome.

Remaining integration acceptance must bind one ordinary package pair to both
physical placements, unchanged model/input/precision/reference requirements,
and the declared memory ceiling. Exercise failed opening, denial, successful
reuse, concurrent attempts, isolated cancellation, contributor loss and restart,
retained weights, and normal application completion. Reploid's requester must
acquire no weights for this integration. Verify selective acquisition separately
from partial GPU materialization. Compare task quality separately against unsplit
execution with identical bytes and settings. No numerical threshold is changed
by a documentation update.

Partial materialization does not establish selective acquisition. Verification
can read an entire shared shard. Tied embeddings can be required by both
partitions. Allocation totals are runtime-owned bytes, not physical VRAM.

## Device allocation budgets and observations

The public partitions entry exposes `configureDeviceMemoryBudget({maxBytes})`
and `inspectDeviceMemory()`. Configure the host's positive byte ceiling before
opening a model (`null` explicitly disables the ceiling). Every GPUBuffer on
that Doppler device counts, including direct weight/cache allocations, pooled
buffers, uniforms, staging and loading temporaries. Allocation fails before the
native call when it would exceed the ceiling. Changing an active finite ceiling
requires releasing the existing allocations. This is a GPUBuffer allocation
budget, not physical VRAM, process RSS, driver, shader or JavaScript heap accounting.

Snapshots retain live and peak bytes, denied allocations, and labeled current
allocations. Resident preparation identifies loaded weight buffers; attention and
recurrent state labels remain separately visible. Other allocations include RoPE,
fused weights and reusable temporaries; the detailed labels distinguish them.
Resident sessions release the unused opening KV cache; every attempt still owns
its independent cache and recurrent state.

Partition steps report wall durations for encoding, submission/wait, upload,
activation/logit readback and sampling. Encoding includes upload; GPU kernel
measurements are a subset of execution, never an additive latency bucket.
`runtime.shared.debug.profiler.enabled` enables timestamp observations when the
device supports them. Absent GPU timestamps stay null. Peer waiting and transport
remain the application's responsibility. Performance observations do not change
precision, token selection, context limits or continuation rules.

Paired operator wrappers preserve explicit Tensor representation (`precision: "two-f32"`, `components: 2`, `elementStrideBytes: 8`). Residual/scale/SiLU entry points do not infer this representation from a global session. Their paired paths reject mixed representations and unsupported clipping or row-split execution before dispatch. Full-shape scale/SiLU outputs preserve the input logical shape; an explicitly shorter prefix has shape `[count]`. Recorded failed outputs remain recorder-owned until cleanup. Operator qualification does not grant partition-plan or whole-model acceptance.
