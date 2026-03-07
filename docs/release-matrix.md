# Release Matrix

Generated: 2026-03-07T20:43:56.088Z
Release: channel=main-snapshot, version=n/a, commit=63a1903a8d0fa960d87ce103420cbc3bbe21881b, dirty=n/a

## Engine Matrix

| Target | Status | Browser WebGPU | Node WebGPU | Headless Harness | Cache Mode Control | Bench Features | Profile Features |
|---|---|---|---|---|---|---:|---:|
| Doppler (`doppler`) | active | yes | yes | yes | yes | 19/19 | 7/8 |
| WebLLM (`webllm`) | active | yes | no | no | no | 5/19 | 0/8 |
| Transformers.js (`transformersjs`) | active | yes | no | yes | yes | 17/19 | 2/8 |
| Ratchet (`ratchet`) | experimental | yes | no | no | no | 5/19 | 0/8 |
| Candle (`candle`) | experimental | no | no | no | no | 4/19 | 0/8 |
| Burn (`burn`) | experimental | no | no | no | no | 4/19 | 0/8 |
| MediaPipe LLM Inference (`mediapipe-llm`) | active | yes | no | no | no | 5/19 | 0/8 |
| Wllama (`wllama`) | active | no | no | no | no | 4/19 | 0/8 |

## Model Coverage

| Doppler Model | In Catalog | Catalog Modes | TJS Mapping | Kernel Path | Surface | Base Dir |
|---|---|---|---|---|---|---|
| `gemma-3-1b-it-wf16-ef16-hf16-f16` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f16a-online` | auto | local |
| `gemma-3-1b-it-wf16-ef16-hf16-f32` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f32a-online` | auto | local |
| `gemma-3-1b-it-wq4k-ef16-hf16` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-q4k-dequant-f16a-online` | auto | local |
| `gemma-3-1b-it-wq4k-ef16-hf16-f32` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-q4k-dequant-f32a-online` | auto | local |
| `gemma-3-270m-it-wf16-ef16-hf16` | no |  | `onnx-community/gemma-3-270m-it-ONNX` |  | auto | local |
| `gemma-3-270m-it-wq4k-ef16-hf16` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` | `gemma3-q4k-dequant-f32a-online` | auto | local |
| `google-embeddinggemma-300m-wq4k-ef16` | yes | embedding |  |  | auto |  |
| `qwen-3-5-0-8b-wq4k-ef16-hf16-f16` | yes | run, translate |  |  | auto |  |
| `qwen-3-5-2b-wq4k-ef16-hf16-f16` | yes | run, translate |  |  | auto |  |
| `translategemma-4b-it-wq4k-ef16-hf16` | yes | run, translate |  |  | auto |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | greedy (t=0) | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-wf16-ef16-hf16-f32 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/lfm2-5-1-2b-p064-d064-t0-k1.compare.json) | lfm2-5-1-2b-instruct-wq4k-ef16-hf16-f32 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | t=1, k=32, p=1 | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.apple-m3pro.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | t=1, k=32, p=1 | Apple M3; metal; darwin; chromium | 2026-02-24 |
| `g3-p256-d128-t0-k1` | not captured | 256 | 128 | greedy (t=0) | not captured | not captured |
| `g3-p512-d128-t0-k1` | not captured | 512 | 128 | greedy (t=0) | not captured | not captured |
| `g3-p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

