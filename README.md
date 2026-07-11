# doppler-gpu

[![Build](https://img.shields.io/github/actions/workflow/status/clocksmith/doppler/check-green.yml?branch=main&label=build)](https://github.com/clocksmith/doppler/actions/workflows/check-green.yml)
[![npm version](https://img.shields.io/npm/v/doppler-gpu.svg?label=version)](https://www.npmjs.com/package/doppler-gpu)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/clocksmith/doppler/blob/main/LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/clocksmith/doppler/pulls)

Doppler turns inference engineering into a verifiable search problem. Optimizers
edit RDRR manifests, execution plans, and kernels; parity and benchmark gates
accept or reject each candidate against correctness and speed.

Today engineers use these contracts and gates to tune a JavaScript/WGSL WebGPU
runtime for supported
[RDRR artifacts](https://github.com/clocksmith/doppler/blob/main/docs/rdrr-format.md)
in browser and Node. Doppler runs text generation, embeddings, and reranking
locally, with CLI and OpenAI-compatible server entry points. Bun WebGPU support
is experimental.

**[Try the live demo](https://d4da.com/doppler)** | **[npm](https://www.npmjs.com/package/doppler-gpu)** | **[docs](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)**

```bash
npx doppler-gpu
```

## Evidence

Doppler has lower steady-state inference latency than Transformers.js in each
of the comparable browser WebGPU results indexed below.

![Metal and Vulkan browser WebGPU latency distributions](https://raw.githubusercontent.com/clocksmith/doppler/main/assets/doppler-webgpu-evidence.svg)

For text, faster means lower decode ms/token; for retrieval it means lower
ms/embedding or ms/rerank. Every comparison passes its declared workload
correctness gate. Model loading is separate: Transformers.js loads the Vulkan
embedding and reranker artifacts faster. The
[competition scoreboard](https://github.com/clocksmith/doppler/blob/main/docs/model-competition-scoreboard.md) links every
receipt, and the [methodology](https://github.com/clocksmith/doppler/blob/main/docs/benchmark-methodology.md) defines the gates.

## Why these lanes are faster

Doppler authors the WGSL path and pins it in RDRR. Runtime profiles pin session
cadence. The measured wins come from different mechanisms in different phases:

| Lever | Affected phase | Measured receipt |
| --- | --- | --- |
| Fused Q4_K projection and FFN work removes separate dispatches | Text decode | [Qwen 3.5 0.8B Metal](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260709T154633.json) |
| Fixed `head_dim=128` attention avoids the generic prefill path | Retrieval attention | [embedding](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/embedding_compare_qwen-3-embedding-0-6b-q4k-ehf16-af32_20260709T180853.json), [reranking](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/rerank_compare_qwen-3-reranker-0-6b-q4k-ehf16-af32_20260709T192830.json) |
| Batch-four decode amortizes one readback across four tokens | Submit and map waits | [Qwen 3.5 2B Vulkan](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260707T161623.json) |
| INT4-PLE and Q4_K layouts lower projection traffic | Text decode | [Gemma 4 Vulkan](https://github.com/clocksmith/doppler/blob/main/benchmarks/vendors/results/compare_20260707T170557.json) |

The Gemma 4 comparison uses its declared product-format output policy. It does
not claim exact greedy token parity.

Parity and comparable benchmark gates decide what is retained. The
[challenger framework](https://github.com/clocksmith/doppler/blob/main/docs/local-gpu-challenger-framework.md)
records accepted and rejected trials, and each verdict's receipt binds it to a
model, profile, workload, and device; the
[negative-results findings](https://github.com/clocksmith/doppler/blob/main/docs/developer-guides/16-kernel-performance-optimization.md#what-we-ruled-out)
keep the dead ends.

### Long-term direction

Humans run this loop today. The WGSL-distillation workload is experimental.
Automated kernel generation and autotuning are directions, not shipped product
paths. Doppler aims to reduce the manual steps between a new checkpoint or GPU
and a verified runtime lane. Automated proposals will face the same parity and
benchmark gates.

## How it works

```text
+----------+     +------------------+     +--------------+     +-----------------+     +------------------+
| Contract | --> | Candidate        | --> | Parity gate  | --> | Benchmark gate  | --> | Retain or reject |
| RDRR /   |     | model / plan /   |     | correctness  |     | comparable      |     | receipts /       |
| request  |     | kernel edit      |     | vs reference |     | speed vs prior  |     | findings         |
+----------+     +------------------+     +--------------+     +-----------------+     +--------+---------+
                                                                                               |
                                                                            next search <------+
```

The full resolve, load, bind, dispatch, and readback flow lives in the
[architecture](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md)
document. Unsupported paths fail closed. Doppler owns artifact and execution
contracts; applications own policy.

New model families need RDRR conversion and may need tokenizer, graph, or kernel
support.

[Ouroboros/Reploid](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md#optional-ouroborosreploid-integration)
keeps orchestration above this boundary.
[Program Bundles](https://github.com/clocksmith/doppler/blob/main/docs/integration/program-bundle.md)
preserve program identity for downstream backends.

## Quick start

### Browser

The live demo runs locally and works offline after its first model download.

### CLI

```bash
npx doppler-gpu "Summarize WebGPU in one sentence"
npx doppler-gpu --model qwen3-0.8b --prompt "Write a haiku about GPUs"
npx doppler-gpu --list-models
```

### Root API

The `dr` facade is the primary app API. Advanced APIs use package subpaths.

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

```bash
npx doppler-serve --model qwen3-0.8b --port 8080
```

Point an OpenAI client at `http://localhost:8080/v1`:

```js
import OpenAI from 'openai';
const client = new OpenAI({ baseURL: 'http://localhost:8080/v1', apiKey: 'unused' });
const response = await client.chat.completions.create({
  model: 'qwen3-0.8b',
  messages: [{ role: 'user', content: 'Hello' }],
});
```

Registry IDs resolve to hosted RDRR artifacts from `Clocksmith/rdrr` by default. See the [Root API guide](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md).

## Start here

| Reader | Entry points |
| --- | --- |
| Application developers | [Getting started](https://github.com/clocksmith/doppler/blob/main/docs/getting-started.md), [Root API](https://github.com/clocksmith/doppler/blob/main/docs/api/root.md), and the [OpenAI-compatible server](#openai-compatible-server) |
| Model integrators | [RDRR format](https://github.com/clocksmith/doppler/blob/main/docs/rdrr-format.md), [model support](https://github.com/clocksmith/doppler/blob/main/docs/model-support-matrix.md), and [Program Bundles](https://github.com/clocksmith/doppler/blob/main/docs/integration/program-bundle.md) |
| Runtime and kernel engineers | [Architecture](https://github.com/clocksmith/doppler/blob/main/docs/architecture.md), [kernel optimization](https://github.com/clocksmith/doppler/blob/main/docs/developer-guides/16-kernel-performance-optimization.md), and the [challenger framework](https://github.com/clocksmith/doppler/blob/main/docs/local-gpu-challenger-framework.md) |
| Evidence reviewers | [Competition scoreboard](https://github.com/clocksmith/doppler/blob/main/docs/model-competition-scoreboard.md), [benchmark methodology](https://github.com/clocksmith/doppler/blob/main/docs/benchmark-methodology.md), and [release matrix](https://github.com/clocksmith/doppler/blob/main/docs/release-matrix.md) |

The [docs index](https://github.com/clocksmith/doppler/blob/main/docs/INDEX.md)
owns the complete model, subsystem, API, architecture, and operator inventory.

## Environment requirements

WebGPU is required. Use a current Chromium browser; Node installs the `webgpu`
provider as an optional dependency.

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
