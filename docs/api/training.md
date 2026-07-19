# Training API

## Import

```js
import {
  getTrainingCapabilities,
  loadTrainingWorkloadPack,
  trainSftLoRA,
} from 'doppler-gpu/training';
```

`doppler-gpu/training` is the public experimental library for Doppler's
completion-masked SFT/LoRA engine. The operator lifecycle runs in Node. The
export also exposes the WebGPU autograd, optimizer, loss, checkpoint, and LoRA
serialization primitives used by the native runner.

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
  backend: 'external',
  loadedWorkload,
  causalLmTrainer,
  runRoot: './reports/my-run',
});
```

The native full-graph backend currently supports the declared native fixtures,
including `gemma-3-270m-it-f16-af32`. Registered packed Q4K Gemma and Qwen
workloads require `external` plus `causalLmTrainer` or the workload's explicit
`pipeline.trainer` module. Registration alone never implies native backward
support.

## Output

Successful causal-LM runs produce a safetensors LoRA weight file, an RDRR LoRA
adapter manifest, evaluation reports, lineage, and run receipts. These prove
the executed training contract; capability or quality promotion still depends
on the workload's independent evaluation and quality gates.

See [Training Handbook](../training-handbook.md) for workload schemas, command
equivalents, artifact layout, and claim boundaries.
