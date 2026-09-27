# Bounded partition numerical diagnostic

This batch establishes an internal GPU layer-execution foundation. It does not
implement or qualify Reploid's complete resident-session API. See the
[implementation and remaining acceptance](../../../docs/distribution/resident-partition-execution.md).

## Inspected and changed

Started from clean Doppler `344c1e51b699094b1bdcaacb6c26d3a3fcf6d3d6` and
Reploid `7e5011cd`. Read the consumer handoff, exact session/step types,
conformance harness, partial loader, model lifecycle, cache policy and attention
indexing, existing prefill/decode/logits paths, Capsule composition and artifact
verification, and applicable charters and style guides.

- Partial pipelines allocate only assigned layer caches while deriving cache
  layout from the complete model. A range view translates original indices at
  the cache boundary and preserves cache subclass identity.
- The internal executor runs embedding on A, the assigned layers on each side,
  and the final normalization/head on B. Activations cross the boundary as owned
  bytes. Existing kernels retain model math and precision choices. Execution
  runs under the existing serialized pipeline lease.
- Unsupported architectural/cache dependencies and precision transitions fail
  explicitly. Temporary resource cleanup covers cancellation and failure.
- A physical decode check exposed a mutable configuration returned when matmul
  constants specialize dispatch geometry. That independent correctness repair
  freezes the specialized configuration and preserves the canonical uniform
  layout. The original failed run remains in `initial-physical-failure.log`.
- Reploid's handoff now links this evidence and records the remaining decoder
  finalization contract. No Reploid transport, grants, chat, browser fixture,
  dependency lock or vendored runtime archive changed.

## Executed evidence

`receipt.json` records final source-file hashes, source bases, environment,
commands and results. `physical-result.json` records the model manifest,
tokenizer and shard hashes, prompt IDs, partition plan, runtime override,
per-step comparisons, layer ownership and allocation observations. Raw command
logs accompany those records.

The physical test uses `gemma-3-270m-it-f16-af32`, split after layer 8, with f32
activations and f16 KV. It compares the entire next-token logit vector at
prefill and two decode steps with the ordinary unsplit pipeline. The declared
absolute tolerance is `1e-4` and the existing comparison contract requires
cosine similarity at least `0.9999`. The reference and split paths select the
same greedy tokens and the observed maximum absolute difference is zero on
this fixture.

Failure checks cover invalid token IDs, malformed activation length,
cancellation before dispatch, recording failure, output allocation failure,
cancellation during readback and cancellation after submission. Active pooled
buffer counts return to the pre-check value. This observation excludes retained
weight allocations, direct device allocations and physical VRAM.

Reproduce the physical check from the Doppler root:

```bash
DOPPLER_PARTITION_MODEL_DIR=models/local/gemma-3-270m-it-f16-af32 \
  node tests/integration/partition-layers-physical.test.js
```

The test skips without that explicit model-directory input; a normal unit-suite
pass does not stand in for its retained physical run. No benchmark speed or
latency claim is made.

## Remaining limitations and skipped work

- Both partitions and the reference ran in one Node process on one physical GPU.
  This is not browser, network, multi-machine or capacity-pooling qualification.
- The reference is Doppler's normal unsplit execution using the same GPU kernels,
  not an independent upstream model oracle. Agreement localizes partitioning
  errors; it does not prove the underlying kernels/model conversion correct.
- Only the identified model, prompt, split and short continuation were exercised.
  No claim extends to f16 activation execution, other models, arbitrary split
  points, long context, non-greedy sampling or generation quality.
- Partial materialization does not prove selective acquisition. The test hashes
  every source artifact, and the loader may verify/read shared shards in full.
  Runtime artifact-read costs are not separately measured here. Tied embedding
  dependencies can duplicate resident weight allocations across A and B.
- The resident factory, verified Capsule partition construction, attempt-specific
  state/continuations, sampling/stop binding and Reploid numerical conformance
  harness remain the next increment. The current low-level executor must not be
  represented as that factory. Retire attempt state after execution failure.
- Reploid's per-request `maxTokens` is not currently carried in each runtime step.
  Decoder finalization at a shorter request limit needs an explicit integration
  contract before full text/stopping parity can be accepted.
- No browser fixture was replaced, no accepted search release was repackaged,
  and no deployment, model publication or external adopter claim was attempted.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.inference`, `doppler.runtime-source.gpu.kernels`.
Intent: preserved.
Acceptance evidence: `receipt.json`, `physical-result.json`, executed logs.
Boundary effects: internal partition execution and partial cache allocation;
Reploid documentation only. Public resident-session availability is unchanged.
