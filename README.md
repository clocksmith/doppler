# doppler-gpu

[![Build](https://img.shields.io/github/actions/workflow/status/clocksmith/doppler/check-green.yml?branch=main&label=build)](https://github.com/clocksmith/doppler/actions/workflows/check-green.yml)
[![npm version](https://img.shields.io/npm/v/doppler-gpu.svg?label=version)](https://www.npmjs.com/package/doppler-gpu)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/clocksmith/doppler/blob/main/LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/clocksmith/doppler/pulls)

JavaScript and WGSL WebGPU inference for browser and Node, with CLI and
OpenAI-compatible local server entry points. Doppler loads sharded
[RDRR model artifacts](./docs/rdrr-format.md) for text generation, embeddings,
and reranking. Bun WebGPU support is experimental.

**[Try the live demo](https://d4da.com/doppler)** | **[npm](https://www.npmjs.com/package/doppler-gpu)** | **[docs](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)**

## Current evidence

Doppler is faster than Transformers.js on every model shown below, across
Apple Metal and AMD Vulkan.

![Metal and Vulkan browser WebGPU latency distributions](https://raw.githubusercontent.com/clocksmith/doppler/main/assets/doppler-webgpu-evidence.svg)

Metal evidence: [Qwen 3.5 0.8B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260709T154633.json) ·
[Qwen 3 Embedding 0.6B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260709T180853.json) ·
[Qwen 3 Reranker 0.6B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260709T192830.json) ·
[Qwen 3.5 2B paired run](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/qwen35-2b-metal-paired-p256-d512-20260710.json) ·
[runtime profile](https://github.com/clocksmith/doppler/blob/main/src/config/runtime/profiles/qwen-3-5-2b-metal-parity.json)

Vulkan evidence: [Qwen 3.5 0.8B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260707T153509.json) ·
[Qwen 3 Embedding 0.6B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260710T011455.json) ·
[Qwen 3 Reranker 0.6B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260710T014450.json) ·
[Qwen 3.5 2B](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260707T161623.json) ·
[Gemma 4](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260707T170557.json)

The links include output checks, load time, and hardware details. Qwen 3.5 2B
Metal used 20 paired runs and a local Doppler model artifact. Transformers.js
still loads the Vulkan embedding and reranker models faster. See the
[methodology](https://github.com/clocksmith/doppler/blob/main/docs/benchmark-methodology.md)
and [full results](https://github.com/clocksmith/doppler/blob/main/docs/release-matrix.md).

## How it works

```text
registry ID / model URL
          |
          v
+----------------------+    +----------------------+
| RDRR manifest        |--->| verified shards      |
| model + tokenizer    |    | OPFS / disk cache    |
| session + execution  |    +----------+-----------+
+----------------------+               |
                                       |
prompt / documents                     v
        +--------------------->+----------------------+
                               | JavaScript runtime   |
                               | prefill / decode / KV|
                               +----------+-----------+
                                          |
                                          v
                               +----------------------+
                               | WGSL / WebGPU        |
                               | selected kernels     |
                               +----------+-----------+
                                          |
                                          v
                              text / embeddings / scores
```

The manifest and runtime config select dtype and kernel paths before execution.
Unsupported paths fail closed.

## Quick start

### Browser

Use the live demo link above. It runs entirely in the browser with no server
required. Models load into the browser cache and work offline after the first
download.

### CLI

```bash
npx doppler-gpu
```

Downloads the default quickstart model, runs a local prompt, and prints the answer.
Node quickstart artifacts are cached in `~/.cache/doppler-gpu/models` after the
first run; set `DOPPLER_QUICKSTART_CACHE_DIR` to move the cache or
`DOPPLER_QUICKSTART_CACHE=0` to disable it.

```bash
npx doppler-gpu "Summarize WebGPU in one sentence"
npx doppler-gpu --model qwen3-0.8b --prompt "Write a haiku about GPUs"
npx doppler-gpu --list-models
```

### Root API

The `dr` facade is the primary app-facing API. `doppler` remains a compatibility
alias. Advanced APIs live on explicit package subpaths.

```js
import { dr } from 'doppler-gpu';

// Stream tokens
const model = await dr.load('qwen3-0.8b');
for await (const token of model.generate('Describe WebGPU briefly')) {
  process.stdout.write(token);
}

// One-shot
const text = await model.generateText('Explain WebGPU in one sentence');
```

### OpenAI-compatible server

For existing apps, SDKs, and eval stacks that speak the OpenAI protocol:

```bash
npx doppler-serve --model qwen3-0.8b --port 8080
```

Then point any OpenAI client at `http://localhost:8080/v1`:

```js
import OpenAI from 'openai';
const client = new OpenAI({ baseURL: 'http://localhost:8080/v1', apiKey: 'unused' });
const response = await client.chat.completions.create({
  model: 'qwen3-0.8b',
  messages: [{ role: 'user', content: 'Hello' }],
});
```

This compatibility bridge uses the same runtime contract as the browser and Node APIs.

Registry IDs resolve to hosted RDRR artifacts from `Clocksmith/rdrr` by default. See the [Root API guide](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md).

## Support

The primary proof surface is the hosted browser demo, root `dr` API, quickstart
CLI, OpenAI-compatible local server, and the hosted Qwen registry lanes below.

| Registry alias | Artifact ID | Task |
| --- | --- | --- |
| `qwen3-0.8b` | `qwen-3-5-0-8b-q4k-ehaf16` | Text generation |
| `qwen3-embedding-0.6b` | `qwen-3-embedding-0-6b-q4k-ehf16-af32` | Embeddings |
| `qwen3-reranker-0.6b-q4k` | `qwen-3-reranker-0-6b-q4k-ehf16-af32` | Reranking |

Browser and Node are mainline runtime surfaces. Bun WebGPU is experimental.
Use the [model support matrix](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)
for verified models and the
[subsystem support matrix](https://github.com/clocksmith/doppler/blob/main/docs/subsystem-support-matrix.md)
for public, experimental, and internal-only APIs.

## Model roadmap

Current model priorities and promotion state live in the
[model roadmap](https://github.com/clocksmith/doppler/blob/main/docs/model-roadmap.md).
Exact registry IDs, runtime verification, and benchmark claims remain in the
[model support inventory](https://github.com/clocksmith/doppler/blob/main/docs/model-support-inventory.md)
and [release matrix](https://github.com/clocksmith/doppler/blob/main/docs/release-matrix.md).

## Documentation

- npm quickstart: run `npx doppler-gpu --help`
- Docs index (canonical navigation): [docs/INDEX.md](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)
- First-run workflow: [docs/getting-started.md](https://github.com/clocksmith/doppler/blob/main/docs/getting-started.md)
- CLI reference: [docs/cli.md](https://github.com/clocksmith/doppler/blob/main/docs/cli.md)
- Runtime config contract: [docs/config.md](https://github.com/clocksmith/doppler/blob/main/docs/config.md)
- Architecture: [docs/architecture.md](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md)
- Model roadmap: [docs/model-roadmap.md](https://github.com/clocksmith/doppler/blob/main/docs/model-roadmap.md)
- Model support matrix: [docs/model-support-matrix.md](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md)

## Environment requirements

- WebGPU is required.
- **Browser**: Current Chromium browsers with WebGPU enabled, including Chrome and Edge.
  WebGPU shipped in Chrome/Edge 113+. Firefox and Safari support varies.
- **Node**: Requires a WebGPU provider (`webgpu` npm package). Installed automatically as an optional dependency.

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
