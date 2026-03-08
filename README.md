# @simulatte/doppler

Inference and training on raw WebGPU. Pure JS + WGSL.

**[Live Demo](https://d4da.com)** · **[npm](https://www.npmjs.com/package/@simulatte/doppler)** · **[simulatte.world](https://simulatte.world)**

## Install

```bash
npm install @simulatte/doppler
```

## Quick start

```js
import { doppler } from '@simulatte/doppler';

const model = await doppler.load('gemma3-270m');

for await (const token of model.generate('Hello, world')) {
  process.stdout.write(token);
}
```

Registry IDs resolve to hosted RDRR artifacts from `Clocksmith/rdrr` by default. Tokens stream from a native `AsyncGenerator`. See [more examples](#more-examples) below or the canonical [Root API guide](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md).

## Why Doppler

**JS → WGSL → WebGPU.** Direct JavaScript orchestration into native WebGPU kernels, avoiding ONNX runtimes, WASM blobs, and bridge layers.

**`for await` streaming.** Generation uses a native `AsyncGenerator` that fits normal app control flow.

**LoRA hot-swap.** Swap adapters at runtime without reloading the base model.

**Independent model instances.** Run multiple models concurrently. Each owns its pipeline, buffers, and KV cache.

## Evidence

![Phase-latency comparison on one workload across models](https://raw.githubusercontent.com/clocksmith/doppler/main/benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

Snapshot artifacts:
- [g3-1b-p064-d064-t0-k1.compare.json](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json)
- [lfm2-5-1-2b-p064-d064-t0-k1.compare.json](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/fixtures/lfm2-5-1-2b-p064-d064-t0-k1.compare.json)

## Under the hood

- Sharded weight loading via OPFS moves multi-GB weights into VRAM without blocking the main thread.
- Quantized inference paths (Q4K, Q8, F16) support practical model sizes on consumer GPUs.
- Kernel hot-swap between prefill and decode paths.
- Config-driven runtime keeps presets, kernel-path selection, and sampling explicit.
- Reproducible benchmarks expose deterministic knobs and auditable kernel traces.

## More examples

```js
// Non-streaming
const text = await model.generateText('Explain WebGPU in one sentence');

// Load with progress logging
const modelWithProgress = await doppler.load('gemma3-270m', {
  onProgress: ({ message }) => console.log(`[doppler] ${message}`),
});

// Chat
const reply = await model.chatText([
  { role: 'user', content: 'Write a dispatch that outruns its own light cone' },
]);

// LoRA hot-swap
await model.loadLoRA('https://example.com/adapters/oneshift-twoshift-redshift-blueshift/manifest.json');

// Convenience shorthand (caches model automatically)
for await (const token of doppler('Hello', { model: 'gemma3-270m' })) {
  process.stdout.write(token);
}
```

## Documentation

- Docs index (canonical navigation): [docs/INDEX.md](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)
- First-run workflow: [docs/getting-started.md](https://github.com/clocksmith/doppler/blob/main/docs/getting-started.md)
- Runtime config contract: [docs/config.md](https://github.com/clocksmith/doppler/blob/main/docs/config.md)
- Architecture: [docs/architecture.md](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md)
- Generated model support table: [docs/model-support-matrix.md](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)

Current model support is generated from the catalog and conversion registry.
See [docs/model-support-matrix.md](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md) for the canonical verified, failing, and unverified status table.

## Environment requirements

- WebGPU is required.
- Supported runtimes: WebGPU-capable browsers, or Node with a WebGPU provider.
- Chrome / Edge 113+ supported.
- Firefox support varies (typically behind a flag).
- Safari support is evolving.

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
