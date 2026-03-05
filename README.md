# @simulatte/doppler

Inference and training on raw WebGPU. Pure JS + WGSL.

**[Live Demo](https://d4da.com)** · **[npm](https://www.npmjs.com/package/@simulatte/doppler)** · **[simulatte.world](https://simulatte.world)**

## Install

```bash
npm install @simulatte/doppler
```

## Quick Start

```js
import { doppler } from '@simulatte/doppler';

const model = await doppler.load('gemma-3-1b');

for await (const token of model.generate('Hello, world')) {
  process.stdout.write(token);
}
```

Tokens stream from a native `AsyncGenerator`. See [more examples](#more-examples) below or the full [API contract](docs/doppler-api-contract.md).

## Why Doppler

**JS → WGSL → WebGPU.** One hop to the GPU. No ONNX runtime, no WASM blob, no bridge layer.

**`for await` streaming.** Not callbacks. Not a `TextStreamer` class. A loop.

**LoRA hot-swap.** Swap adapters at runtime without reloading the base model.

**Independent model instances.** Run multiple models concurrently. Each owns its pipeline, buffers, and KV cache.

## Under the Hood

- Sharded weight loading via OPFS. Gigabytes into VRAM without blocking the main thread.
- Quantized inference: Q4K, Q8, F16. Real models on consumer GPUs.
- Kernel hot-swap between prefill and decode paths.
- Config-driven runtime. Presets, kernel path selection, and sampling are policy, not code.
- Reproducible benchmarks with deterministic knobs and auditable kernel traces.

## Browser Support

- Chrome / Edge 113+ (WebGPU required)
- Firefox (behind flag, WebGPU support varies)
- Safari (WebGPU support in progress)

---

## Evidence

![Phase-latency comparison on one workload across models](benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

Snapshot artifacts:
- [g3-1b-p064-d064-t0-k1.compare.json](benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json)
- [lfm2-5-1-2b-p064-d064-t0-k1.compare.json](benchmarks/vendors/fixtures/lfm2-5-1-2b-p064-d064-t0-k1.compare.json)

## More Examples

```js
// Non-streaming
const text = await model.generateText('Explain WebGPU in one sentence');

// Chat
const reply = await model.chatText([
  { role: 'user', content: 'Write a dispatch that outruns its own light cone' },
]);

// LoRA hot-swap
await model.loadLoRA('oneshift-twoshift-redshift-blueshift');

// Convenience shorthand (caches model automatically)
for await (const token of doppler('Hello', { model: 'gemma-3-1b' })) {
  process.stdout.write(token);
}
```

## Documentation

- Docs index (canonical navigation): [docs/INDEX.md](docs/INDEX.md)
- First-run workflow: [docs/getting-started.md](docs/getting-started.md)
- Runtime config contract: [docs/config.md](docs/config.md)
- Architecture: [docs/architecture.md](docs/architecture.md)

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
