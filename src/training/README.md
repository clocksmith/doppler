# Training (Engine)

Purpose: Training primitives for Doppler, including autograd, losses, and adapter export.

## Scope

- Autograd tape, backward kernels, and loss scaling.
- Training runner, optimizers, datasets, and LoRA export helpers.

This directory provides training primitives for Doppler:

- Autograd tape + backward kernels
- Cross-entropy loss + gradient clipping
- Dynamic loss scaling for stability
- Training runner for epochs/steps + metrics
- LoRA adapter export
- Dataset adapters (JSONL, text pairs, token batches)

The training engine is meant to be driven by a higher-level app or agent
(Reploid, gamma, or your own UI). It does not decide *when* to train or *what*
data to use.

## Quick Start (Programmatic)

```js
import {
  TrainingRunner,
  AdamOptimizer,
  crossEntropyLoss,
  clipGradients,
} from './index.js';

const runner = new TrainingRunner(config, {
  optimizer: new AdamOptimizer(config),
  crossEntropyLoss,
  clipGradients,
});

const metrics = await runner.run(model, dataset, {
  epochs: 1,
  batchSize: 1,
  logEvery: 1,
});
```

`model` must implement:

```js
{
  forward: (input, tape) => Promise<Tensor>,
  loraParams: () => Tensor[]
}
```

## Attention Cache vs Recompute

The autograd tape supports two modes:

- **Cache softmax (default):** pass a softmax tensor into the attention record.
- **Recompute:** omit softmax and set `recomputeForward: true` in the record options.

Helper for caching (CPU path, used only for training):

```js
import { buildAttentionSoftmaxCache } from './index.js';
```

If `recomputeForward` is true, the backward path recomputes softmax on CPU
to avoid storing attention weights in VRAM. Use `recordAttentionForward`
to wire the config into a training forward pass:

```js
import { recordAttentionForward } from './index.js';

const { output, softmax } = await recordAttentionForward(q, k, v, config, tape, {
  seqLen,
  numHeads,
  headDim,
});
```

## Loss Scaling & Stability

Dynamic loss scaling is configured via `training.lossScaling`:

```
training: {
  lossScaling: {
    enabled: true,
    initialScale: 1024,
    growthInterval: 2000,
  }
}
```

The runner unscales grads before clipping and optimizer steps, and can detect
NaN/Inf overflow via readback.

## Dataset Adapters

Use the dataset helpers to ingest JSONL or text pairs:

```js
import { datasets } from './index.js';

const records = await datasets.loadJsonl('/data/traces.jsonl');
const pairs = datasets.reploidTracesToTextPairs(records);
const samples = await datasets.tokenizeTextPairs(tokenizer, pairs, { maxLength: 256 });
const tokenBatch = datasets.buildTokenBatch(samples);
const batch = datasets.createTokenBatchTensors(tokenBatch);
```

Note: token ID tensors are stored in GPU buffers but wrapped as `Tensor` with
`dtype: 'f32'`. The loss kernels only read the raw buffer as `u32`.

## Adapter Export

Export LoRA adapters for runtime loading:

```js
import { exportLoRAAdapter } from './index.js';

const { manifest, json } = await exportLoRAAdapter({
  id: 'toolcall-v1',
  name: 'Toolcall v1',
  baseModel: 'gemma-3-1b',
  rank: 16,
  alpha: 32,
  targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
  tensors: [/* ... */],
});
```

Format reference:
- `docs/spec/RDRR_LORA_FORMAT.md`

Optional GGUF conversion helper: removed in browser-only migration.

## Tests

Run training correctness tests via the browser harness (`tests/training/browser/test-page.js`). This
includes kernel numerics, gradient checks, parity fixtures, and leak/perf checks.

## Training Harness

Open the unified harness in the browser with `runtime.shared.harness.mode = "training"` to run a toy training loop.
