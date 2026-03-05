# @simulatte/doppler

Browser-native inference engine for local AI workloads.

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

That's it. Streaming is the default. See [more examples](#more-examples) below or the full [API contract](docs/doppler-api-contract.md).

## Features

- WebGPU-accelerated inference in browser and Node — no WASM bridge
- Native `for await` streaming — not callbacks
- Sharded weight loading via OPFS
- LoRA adapter hot-swap at runtime
- Quantized model support (Q4K, Q8, F16)
- Multi-model with independent instances
- Kernel hot-swap (prefill/decode paths)
- Reproducible benchmark tooling
- Auditable kernel execution tracing

## Browser Support

- Chrome / Edge 113+ (WebGPU required)
- Firefox (behind flag, WebGPU support varies)
- Safari (WebGPU support in progress)

---

## Evidence

Lower is better, comparing per-phase latency by workload.

![Phase-latency comparison on one workload across models](benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

Snapshot artifact:
- [g3-p064-d064-t0-k1.apple-m3pro.compare.json](benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json)

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
