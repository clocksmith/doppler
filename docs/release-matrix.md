# Release Matrix

Generated: 2026-03-19T14:08:42.459Z
Release: channel=main-snapshot, version=0.1.8, commit=7199e9410136419e8ae5b1c39db8452867eda269, dirty=yes

## Engine Matrix

| Target | Status | Browser WebGPU | Node WebGPU | Headless Harness | Cache Mode Control | Bench Features | Profile Features |
|---|---|---|---|---|---|---:|---:|
| Doppler (`doppler`) | active | yes | yes | yes | yes | 19/21 | 7/8 |
| Doppler (@simulatte/webgpu) (`doppler-simulatte`) | active | no | yes | no | no | 16/21 | 7/8 |
| Doppler (webgpu npm) (`doppler-webgpu-npm`) | active | no | yes | no | no | 16/21 | 7/8 |
| Doppler (Bun WebGPU) (`doppler-bun`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Doppler (Deno WebGPU) (`doppler-deno`) | experimental | no | no | no | no | 14/21 | 0/8 |
| WebLLM (`webllm`) | active | yes | no | no | no | 5/21 | 0/8 |
| Transformers.js (`transformersjs`) | active | yes | no | yes | yes | 17/21 | 2/8 |
| Transformers.js v4 (@simulatte/webgpu) (`transformersjs-simulatte`) | experimental | no | yes | no | no | 14/21 | 0/8 |
| Transformers.js v4 (webgpu npm) (`transformersjs-webgpu-npm`) | experimental | no | yes | no | no | 14/21 | 0/8 |
| Transformers.js v4 (Bun WebGPU) (`transformersjs-bun`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Transformers.js v4 (Deno WebGPU) (`transformersjs-deno`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Ratchet (`ratchet`) | experimental | yes | no | no | no | 5/21 | 0/8 |
| Candle (`candle`) | experimental | no | no | no | no | 4/21 | 0/8 |
| Burn (`burn`) | experimental | no | no | no | no | 4/21 | 0/8 |
| MediaPipe LLM Inference (`mediapipe-llm`) | active | yes | no | no | no | 5/21 | 0/8 |
| Wllama (`wllama`) | active | no | no | no | no | 4/21 | 0/8 |

## Model Coverage

| Doppler Model | In Catalog | Catalog Modes | TJS Mapping | Kernel Path | Surface | Base Dir |
|---|---|---|---|---|---|---|
| `gemma-3-1b-it-f16-af32` | yes | run, translate | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f32a-online` | auto | local |
| `gemma-3-1b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-q4k-dequant-f32a-online` | auto | local |
| `gemma-3-270m-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` | `gemma3-q4k-dequant-f16a-online` | auto | local |
| `google-embeddinggemma-300m-q4k-ehf16-af32` | yes | embedding | `onnx-community/embeddinggemma-300m-ONNX` |  | auto | local |
| `lfm2-5-1-2b-instruct-q4k-ehf16-af32` | yes | run, translate | `LiquidAI/LFM2.5-1.2B-Instruct-ONNX` | `lfm2-q4k-dequant-f32a-online` | auto | local |
| `qwen-3-5-0-8b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-0.8B-ONNX` |  | auto | local |
| `qwen-3-5-2b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-2B-ONNX` |  | auto | local |
| `translategemma-4b-1b-enes-q4k-ehf16-af32` | yes | run, translate |  | `gemma3-q4k-dequant-f32a-online` | auto | local |
| `translategemma-4b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/translategemma-text-4b-it-ONNX` | `gemma3-q4k-dequant-f32w-f32a-online` | auto | local |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json) | Gemma 3 1B Instruct (F16/F32a) (1.88 GiB) | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json) | Gemma 3 1B Instruct (F16/F32a) (1.88 GiB) | 64 | 64 | greedy (t=0) | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json) | Gemma 3 1B Instruct (F16/F32a) (1.88 GiB) | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-02-25 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/lfm2-5-1-2b-p064-d064-t0-k1.compare.json) | LFM 2.5 1.2B Instruct (Q4K/F32a) (814 MiB) | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.compare.json) | Gemma 3 1B Instruct (F16/F32a) (1.88 GiB) | 64 | 64 | t=1, k=32, p=1 | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.apple-m3pro.compare.json) | Gemma 3 1B Instruct (F16/F32a) (1.88 GiB) | 64 | 64 | t=1, k=32, p=1 | Apple M3; metal; darwin; chromium | 2026-02-24 |
| `p256-d128-t0-k1` | not captured | 256 | 128 | greedy (t=0) | not captured | not captured |
| `p512-d128-t0-k1` | not captured | 512 | 128 | greedy (t=0) | not captured | not captured |
| `p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

