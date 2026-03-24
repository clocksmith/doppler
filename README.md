# doppler-gpu

Inference and training on raw WebGPU. Pure JS + WGSL.

**[Try the live demo](https://d4da.com)** | **[npm](https://www.npmjs.com/package/doppler-gpu)** | **[docs](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)**

![Phase-latency comparison on one workload across models](https://raw.githubusercontent.com/clocksmith/doppler/main/benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

## Quick start

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
await model.loadLoRA('https://oneshift-twoshift-redshift-blueshift.com/manifest.json');
```

Registry IDs resolve to hosted RDRR artifacts from `Clocksmith/rdrr` by default. Tokens stream from a native `AsyncGenerator`. See the canonical [Root API guide](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md).

## Why Doppler

**JS → WGSL → WebGPU.** Direct JavaScript orchestration into native WebGPU kernels, avoiding ONNX runtimes, WASM blobs, and bridge layers.

**`for await` streaming.** Generation uses a native `AsyncGenerator` that fits normal app control flow.

**LoRA hot-swap.** Swap adapters at runtime without reloading the base model.

**Independent model instances.** Run multiple models concurrently. Each owns its pipeline, buffers, and KV cache.

## Supported models

All models below are verified with deterministic greedy decoding on WebGPU hardware.
Registry IDs resolve to hosted RDRR artifacts automatically.

| Model | Registry ID | Quant | Params |
| --- | --- | --- | --- |
| Gemma 3 270M IT | `gemma3-270m` | Q4K | 270M |
| Gemma 3 1B IT | `gemma3-1b` | Q4K | 1B |
| TranslateGemma 4B IT | `translategemma-4b-it-q4k-ehf16-af32` | Q4K | 4B |
| EmbeddingGemma 300M | `google-embeddinggemma-300m-q4k-ehf16-af32` | Q4K | 300M |
| Qwen 3.5 0.8B | `qwen-3-5-0-8b-q4k-ehaf16` | Q4K | 0.8B |
| Qwen 3.5 2B | `qwen-3-5-2b-q4k-ehaf16` | Q4K | 2B |
| LFM2.5 1.2B Instruct | `lfm2-5-1-2b-instruct-q4k-ehf16-af32` | Q4K | 1.2B |

Additional model families (Llama 3, DeepSeek, Gemma 4 MoE, Mixtral, and others) have conversion
configs ready but are not yet cataloged. See the full
[model support matrix](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)
for details.

## Under the hood

- Sharded weight loading via OPFS moves multi-GB weights into VRAM without blocking the main thread.
- Quantized inference (Q4K, F16) runs practical model sizes on consumer GPUs.
- Kernel hot-swap between prefill and decode paths with zero graph recompilation.
- Config-driven runtime with explicit profiles, kernel-path selection, and sampling.

## Documentation

- Docs index (canonical navigation): [docs/INDEX.md](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)
- First-run workflow: [docs/getting-started.md](https://github.com/clocksmith/doppler/blob/main/docs/getting-started.md)
- CLI reference: [docs/cli.md](https://github.com/clocksmith/doppler/blob/main/docs/cli.md)
- Runtime config contract: [docs/config.md](https://github.com/clocksmith/doppler/blob/main/docs/config.md)
- Architecture: [docs/architecture.md](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md)
- Model support matrix: [docs/model-support-matrix.md](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)

## Environment requirements

- WebGPU is required.
- Supported runtimes: WebGPU-capable browsers, or Node with a WebGPU provider.
- Chrome / Edge 113+ supported.
- Firefox support varies (typically behind a flag).
- Safari support is evolving.

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
