# Release Matrix

Generated: 2026-03-03T20:36:43.233Z
Release: channel=main-snapshot, version=n/a, commit=5b1a7f3fdc8ea92998c54e535b585ef5a5e00d33, dirty=yes

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
| `gemma-3-1b-it-wf16-ef16-hf16` | no |  | `onnx-community/gemma-3-1b-it-ONNX-GQA` | `gemma3-f16-fused-f32a-online` | local |
| `gemma-3-270m-it-wf16-ef16-hf16` | no |  | `onnx-community/gemma-3-270m-it-ONNX` |  | curated |
| `gemma-3-270m-it-wq4k-ef16-hf16-f32` | yes | run, translate |  |  |  |
| `google-embeddinggemma-300m-wq4k-ef16` | yes | embedding |  |  |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | greedy (t=0) | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.apple-m3pro.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | t=1, k=32, p=1 | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; vulkan; linux; chromium | 2026-02-25 |
| [`g3-p064-d064-t1-k32`](../benchmarks/vendors/fixtures/g3-p064-d064-t1-k32.apple-m3pro.compare.json) | gemma-3-1b-it-wf16-ef16-hf16 | 64 | 64 | t=1, k=32, p=1 | Apple M3; metal; darwin; chromium | 2026-02-24 |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)
- [compare_warm-opfs_parity_gemma3-1b-f32_20260303_phases.svg](../benchmarks/vendors/results/compare_warm-opfs_parity_gemma3-1b-f32_20260303_phases.svg)
- [compare_warm-opfs_parity_gemma3-1b-f32_lfm2.5-1.2b_20260303_phases.svg](../benchmarks/vendors/results/compare_warm-opfs_parity_gemma3-1b-f32_lfm2.5-1.2b_20260303_phases.svg)
- [compare_warm-opfs_parity_lfm2.5-1.2b_20260303_phases.svg](../benchmarks/vendors/results/compare_warm-opfs_parity_lfm2.5-1.2b_20260303_phases.svg)

