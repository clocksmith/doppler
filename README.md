# doppler-gpu

Browser-native inference on raw WebGPU. Pure JS + WGSL.

**[Try the live demo](https://d4da.com)** | **[npm](https://www.npmjs.com/package/doppler-gpu)** | **[docs](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)**

[![Representative phase-latency comparison across selected workloads](https://raw.githubusercontent.com/clocksmith/doppler/main/benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)](https://github.com/clocksmith/doppler/blob/main/docs/benchmark-methodology.md)

Warm-cache phase-latency comparison on Gemma 3 1B and LFM 2.5 1.2B (MacBook Air M3, 64 prompt / 64 decode tokens, greedy). Doppler.js vs Transformers.js v4. See the [benchmark methodology](https://github.com/clocksmith/doppler/blob/main/docs/benchmark-methodology.md).

## Quick start

### Browser

Open the [live demo](https://d4da.com) — runs entirely in the browser with no server required. Models load into the browser cache and work offline after first download.

### CLI

```bash
npx doppler-gpu
```

Downloads the default quickstart model, runs a local prompt, and prints the answer.

```bash
npx doppler-gpu "Summarize WebGPU in one sentence"
npx doppler-gpu --model qwen3-0.8b --prompt "Write a haiku about GPUs"
npx doppler-gpu --list-models
```

### API

```js
import { doppler } from 'doppler-gpu';

// Stream tokens
const model = await doppler.load('gemma3-270m');
for await (const token of model.generate('Describe WebGPU briefly')) {
  process.stdout.write(token);
}

// One-shot
const text = await model.generateText('Explain WebGPU in one sentence');

// LoRA hot-swap
await model.loadLoRA('https://example.com/adapter/manifest.json');
```

### OpenAI-compatible server

For existing apps, SDKs, and eval stacks that speak the OpenAI protocol:

```bash
npx doppler-serve --model gemma3-270m --port 8080
```

Then point any OpenAI client at `http://localhost:8080/v1`:

```js
import OpenAI from 'openai';
const client = new OpenAI({ baseURL: 'http://localhost:8080/v1', apiKey: 'unused' });
const response = await client.chat.completions.create({
  model: 'gemma3-270m',
  messages: [{ role: 'user', content: 'Hello' }],
});
```

This is a compatibility bridge — the core engine runs identically in the browser or Node.

Registry IDs resolve to hosted RDRR artifacts from `Clocksmith/rdrr` by default. See the [Root API guide](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md).

## Why Doppler

**Browser-native.** Runs entirely in any WebGPU browser tab — no server, no WASM, no native extensions. Models cache in OPFS and work offline.

**JS → WGSL → WebGPU.** Direct JavaScript orchestration into native WebGPU kernels, avoiding ONNX runtimes and bridge layers.

**`for await` streaming.** Generation uses a native `AsyncGenerator` that fits normal app control flow.

**LoRA hot-swap.** Swap adapters at runtime without reloading the base model.

**Independent model instances.** Run multiple models concurrently. Each owns its pipeline, buffers, and KV cache.

## Quickstart-supported models

All models below are verified with deterministic greedy decoding on WebGPU hardware.
These registry IDs resolve to hosted RDRR artifacts automatically from the browser demo,
`npx doppler-gpu`, or `doppler.load(...)`.

| Model | Registry ID | Quant | Size | Family |
| --- | --- | --- | --- | --- |
| Gemma 3 270M IT | `gemma3-270m` | Q4K | 270M | Gemma |
| Gemma 3 1B IT | `gemma3-1b` | Q4K | 1B | Gemma |
| EmbeddingGemma 300M | `embeddinggemma-300m` | Q4K | 300M | Gemma |
| Qwen 3.5 0.8B | `qwen3-0.8b` | Q4K | 0.8B | Qwen |
| Qwen 3.5 2B | `qwen3-2b` | Q4K | 2B | Qwen |

Additional verified models (TranslateGemma 4B, LFM2.5 1.2B) are available with
local artifacts rather than the quickstart registry. Conversion configs exist
for Gemma 4 MoE, Janus, and Sana but are not yet in the quickstart registry.
See the
[model support matrix](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md).

## Under the hood

- Sharded weight loading via OPFS moves multi-GB weights into VRAM without blocking the main thread.
- Quantized inference (Q4K, F16) runs practical model sizes on consumer GPUs.
- Kernel hot-swap between prefill and decode paths with zero graph recompilation.
- Config-driven runtime with explicit profiles, kernel-path selection, and sampling.

## Documentation

- npm quickstart: run `npx doppler-gpu --help`
- Docs index (canonical navigation): [docs/INDEX.md](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)
- First-run workflow: [docs/getting-started.md](https://github.com/clocksmith/doppler/blob/main/docs/getting-started.md)
- CLI reference: [docs/cli.md](https://github.com/clocksmith/doppler/blob/main/docs/cli.md)
- Runtime config contract: [docs/config.md](https://github.com/clocksmith/doppler/blob/main/docs/config.md)
- Architecture: [docs/architecture.md](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md)
- Model support matrix: [docs/model-support-matrix.md](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)

## Environment requirements

- WebGPU is required.
- **Browser**: Current Chromium browsers with WebGPU enabled, including Chrome and Edge.
  WebGPU shipped in Chrome/Edge 113+. Firefox and Safari support varies.
- **Node**: Requires a WebGPU provider (`webgpu` npm package). Installed automatically as an optional dependency.

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
