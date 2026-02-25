# Release Matrix

Generated: 2026-02-25T19:22:12.061Z
Release: channel=main-snapshot, version=n/a, commit=d10a6ac38483bf2579f72fc3290d22ed1c4cbcd1, dirty=yes

## Engine Matrix

| Target | Status | Browser WebGPU | Node WebGPU | Headless Harness | Cache Mode Control | Bench Features | Profile Features |
|---|---|---|---|---|---|---:|---:|
| Doppler (`doppler`) | active | yes | yes | yes | yes | 18/18 | 7/8 |
| WebLLM (`webllm`) | active | yes | no | no | no | 5/18 | 0/8 |
| Transformers.js (`transformersjs`) | active | yes | no | yes | yes | 16/18 | 2/8 |
| Ratchet (`ratchet`) | experimental | yes | no | no | no | 5/18 | 0/8 |
| Candle (`candle`) | experimental | no | no | no | no | 4/18 | 0/8 |
| Burn (`burn`) | experimental | no | no | no | no | 4/18 | 0/8 |
| MediaPipe LLM Inference (`mediapipe-llm`) | active | yes | no | no | no | 5/18 | 0/8 |
| Wllama (`wllama`) | active | no | no | no | no | 4/18 | 0/8 |

## Model Coverage

| Doppler Model | In Catalog | Catalog Modes | TJS Mapping | Kernel Path | Base Dir |
|---|---|---|---|---|---|
| `gemma-3-1b-it-f16-f32a` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f32a-online` | local |
| `gemma-3-270m-it-f16-f32a` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` |  | curated |
| `google-embeddinggemma-300m-wbf16` | yes | embedding |  |  |  |
| `translategemma-4b-it-wq4` | yes | run, translate |  |  |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-f16-f32a | 64 | 64 | greedy (t=0) | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json) | gemma-3-1b-it-f16-f32a | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.compare.json) | gemma-3-1b-it-f16-f32a | 64 | 64 | t=1, k=32, p=1 | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.apple-m3pro.compare.json) | gemma-3-1b-it-f16-f32a | 64 | 64 | t=1, k=32, p=1 | Apple M3; metal; darwin; chromium | 2026-02-24 |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

