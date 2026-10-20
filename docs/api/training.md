# Training API

## Import

```js
import {
  bootstrapNativeTrainingHost,
  getTrainingCapabilities,
  loadNativeQwenTrainingPipeline,
  loadTrainingWorkloadCapsule,
  releaseNativeTrainingHost,
  trainSftLoRA,
} from 'doppler-gpu/training';
```

`doppler-gpu/training` is the public experimental library for Doppler's
completion-masked SFT/LoRA engine. Native WebGPU primitives and the bounded
Qwen runner are available in browsers, Node, and Bun. The Node entry point also
exports workload-file and report orchestration APIs.

## Capability Gate

Load a frozen workload and select a backend explicitly:

```js
const loadedWorkload = await loadTrainingWorkloadCapsule('./workload.json');
const capabilities = getTrainingCapabilities(loadedWorkload.workload);

const backend = capabilities.backends.webgpuNative.supported
  ? 'webgpu_native'
  : 'external';
```

`webgpu_native` means Doppler owns forward, backward, optimization, and adapter
export. `external` means an injected trainer owns gradient execution while
Doppler owns the workload contract, evaluation artifacts, lineage, and runtime
adapter package. Unsupported combinations fail before training starts.

## Run SFT/LoRA

```js
const result = await trainSftLoRA({
  backend: 'webgpu_native',
  loadedWorkload,
  pipeline: loadedQwenPipeline,
  samples: tokenizedCompletionMaskedSamples,
  export: { id: 'my-adapter', name: 'My adapter', weightsPath: 'adapter.safetensors' },
});
```

The native backend supports the declared native fixtures and one exact packed
Q4K production target: the final `down_proj` in
`qwen-3-5-0-8b-q4k-ehaf16`. Its workload must declare
`pipeline.nativeTarget={"module":"down_proj","layer":"last"}` and request
only `down_proj`. Doppler freezes the Q4K base, runs the production Qwen
forward, and optimizes only LoRA A/B matrices with AdamW. Other packed Q4K
targets still require `external`; registration alone never implies native
backward support.

Node and Bun callers should bracket their work with
`bootstrapNativeTrainingHost()` and `releaseNativeTrainingHost()`. Browser
callers use the browser's WebGPU provider directly. All surfaces can load the
same HTTP-hosted RDRR model with `loadNativeQwenTrainingPipeline(modelUrl)`.

## Output

Successful causal-LM runs produce a safetensors LoRA weight file, an RDRR LoRA
adapter manifest, evaluation reports, lineage, and run receipts. These prove
the executed training contract; capability or quality promotion still depends
on the workload's independent evaluation and quality gates.

See [Training Handbook](../training-handbook.md) for workload schemas, command
equivalents, artifact layout, and claim boundaries.

## Community adapter training and sharing

For individual training, create a `NativeQwenLoRATrainer`, run `trainStep`, and
call `exportAdapter({ id, name, version, weightsPath, description, metadata })`.
The export includes safetensors weights, their checksum, a versioned RDRR adapter
manifest, and the exact base-manifest hash. Keep the manifest and weight file
together. Existing adapter loading and Reploid publication remain the consumers;
export alone does not publish, evaluate, or mark an adapter ready.

For synchronous data-parallel training, each participant keeps the same frozen
base model on its own GPU. The coordinator owns the optimizer. Each participant
can describe its local sample without uploading the sample to other participants:

```js
const assignment = workerTrainer.describeTrainingSample(participantId, sample);
// The application collects assignments and sends this round to each worker.
const round = await coordinator.createTrainingRound({
  jobId, roundId, assignments,
});
const contribution = await workerTrainer.computeGradientContribution(
  round, participantId, sample,
);
// After every declared participant completes, send all contributions here.
const receipt = await coordinator.applyGradientContributions(round, contributions);
const checkpoint = await coordinator.createCheckpoint();
```

The application supplies transport, storage, participant selection, deadlines,
and explicit retry. Doppler validates exact base artifact, training settings,
checkpoint, assignment and round identity. A worker restores the round checkpoint
and produces unclipped mean gradients without updating its optimizer. The
coordinator sums those gradients on the GPU, weighted by supervised-token count,
clips once, and performs one AdamW step. Unequal sample lengths therefore have
the same token weighting as a combined batch. Summation order is the declared
assignment order, independent of network arrival order.

Missing, duplicate, foreign and stale contributions are rejected. Invalid
payload sizes and non-finite aggregate gradients cannot update the optimizer.
A replayed completed round is rejected against the coordinator's changed
checkpoint. Concurrent trainer operations are rejected. An exception during a
GPU optimizer update invalidates that trainer: initialize a new trainer from
the retained checkpoint rather than assuming cleanup or update completion.
If a contributor leaves, the coordinator must retain the original checkpoint
and explicitly retry or create a new round; it cannot silently average a subset.

This is gradient aggregation, not averaging separately trained adapters. It
requires the exact native target listed above; it does not enable arbitrary
layers, 2B native training, or a model split across GPUs. Checkpoint and gradient
transfers may expose training information even though raw samples stay local.
Identity checks do not establish that an untrusted contributor computed a good
update. Evaluate a resulting adapter before making it available as a ready model.

The bounded file-transport qualification driver is
`node tools/qualify-community-lora.js --config <job.json>`. It executes one
`prepare`, `contribute`, `apply`, or `verify` action on already-provisioned weights. It is
repository qualification tooling, not a public network service or training UI.
