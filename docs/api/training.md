# Training API

## Import

```js
import {
  bootstrapNativeTrainingHost,
  getTrainingCapabilities,
  loadNativeQwenTrainingPipeline,
  loadTrainingWorkloadPack,
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
const loadedWorkload = await loadTrainingWorkloadPack('./workload.json');
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
