# Release Matrix

Generated: 2026-03-09T00:10:00.344Z
Release: channel=main-snapshot, version=n/a, commit=31d4141aa5d84b63cb70e576193cc629d6a5fe92, dirty=n/a

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
| `gemma-3-1b-it-f16-af32` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f32a-online` | auto | local |
| `gemma-3-1b-it-q4k-ehf16-af32` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-q4k-dequant-f32a-online` | auto | local |
| `gemma-3-270m-it-f16-af32` | no |  | `onnx-community/gemma-3-270m-it-ONNX` |  | auto | local |
| `gemma-3-270m-it-q4k-ehaf16` | no |  | `onnx-community/gemma-3-270m-it-ONNX` | `gemma3-q4k-dequant-f16a-online` | auto | local |
| `gemma-3-270m-it-q4k-ehf16-af32` | yes | run, translate |  |  | auto |  |
| `google-embeddinggemma-300m-q4k-ehf16-af32` | yes | embedding |  |  | auto |  |
| `qwen-3-5-0-8b-q4k-ehaf16` | yes | run, translate |  |  | auto |  |
| `qwen-3-5-2b-q4k-ehaf16` | yes | run, translate |  |  | auto |  |
| `translategemma-4b-it-q4k-ehf16-af32` | yes | run, translate |  |  | auto |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-f16-af32 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-f16-af32 | 64 | 64 | greedy (t=0) | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json) | gemma-3-1b-it-f16-af32 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/lfm2-5-1-2b-p064-d064-t0-k1.compare.json) | lfm2-5-1-2b-instruct-q4k-ehf16-af32 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-03 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.compare.json) | gemma-3-1b-it-f16-af32 | 64 | 64 | t=1, k=32, p=1 | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.apple-m3pro.compare.json) | gemma-3-1b-it-f16-af32 | 64 | 64 | t=1, k=32, p=1 | Apple M3; metal; darwin; chromium | 2026-02-24 |
| `g3-p256-d128-t0-k1` | not captured | 256 | 128 | greedy (t=0) | not captured | not captured |
| `g3-p512-d128-t0-k1` | not captured | 512 | 128 | greedy (t=0) | not captured | not captured |
| `g3-p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

