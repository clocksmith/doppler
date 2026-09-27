# Resident partition execution

The numerical foundation executes the assigned transformer layers with Doppler's
existing GPU kernels. It is internal and is not an `openResidentPartition`
implementation. The public `doppler-gpu/partitions` import remains a contract and
codec surface.

The [Reploid consumer handoff](https://github.com/clocksmith/reploid/blob/main/docs/doppler-partition-handoff.md)
defines the intended resident-session boundary. Doppler owns numerical execution
and model state; Reploid owns grants, placement, transfer, and conversation
coordination. The [local diagnostic](../../reports/resident-partitions/20260927/README.md)
records executed checks and their scope.

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

Supported admission currently requires dense causal incremental attention.
Recurrent layers, MoE, shared KV, per-layer inputs, adapters, multimodal execution,
cross-layer normalization fusion and finiteness fallback transitions are rejected.
The retained numerical evidence covers only the exact artifact and precision
specified in its report.

## Next integration acceptance

1. Construct the dedicated resident factory through verified Capsule acquisition
   and the existing composition root. Preserve signed source, tokenizer, shader,
   registry, release and execution identities; keep the minimal Capsule root
   independent of the executor.
2. Give each full attempt identity its own KV state and bounded continuation.
   Reject replay, identity changes and ordering violations at the runtime entry.
   Close and cancellation must settle operations before freeing attempt state;
   resident weights survive attempt closure.
3. Bind sampling, stopping, incremental decoding and allocation limits during
   opening. Reconcile per-request output limits with decoder finalization: the
   current Reploid step interface does not carry the request's `maxTokens`.
   A runner stopping early must not lose the decoder's pending text.
4. Run Reploid's existing `qualifyDopplerPartitionSessions` harness with real
   residents and an unsplit reference, including multiple attempts and failures.
   Then replace injected browser arithmetic and qualify the installed-package
   two-tab path. Physical multi-machine and capacity-pooling claims remain
   separate acceptance work.

Partial materialization does not establish selective acquisition. Verification
can read an entire shared shard. Tied embeddings can be required by both
partitions. Allocation totals are runtime-owned bytes, not physical VRAM.
