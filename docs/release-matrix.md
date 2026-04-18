# Release Matrix

Generated: 2026-04-18T06:19:41.316Z
Release: channel=main-snapshot, version=0.4.2, commit=66e2fd83dd7832eafa91aaa0252dcff03c1a92fa, dirty=yes

## Engine Matrix

| Target | Status | Browser WebGPU | Node WebGPU | Headless Harness | Cache Mode Control | Bench Features | Profile Features |
|---|---|---|---|---|---|---:|---:|
| Doppler (`doppler`) | active | yes | yes | yes | yes | 19/21 | 7/8 |
| Doppler (webgpu) (`doppler-simulatte`) | active | no | yes | no | no | 16/21 | 7/8 |
| Doppler (webgpu npm) (`doppler-webgpu-npm`) | active | no | yes | no | no | 16/21 | 7/8 |
| Doppler (Bun WebGPU) (`doppler-bun`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Doppler (Deno WebGPU) (`doppler-deno`) | experimental | no | no | no | no | 14/21 | 0/8 |
| WebLLM (`webllm`) | active | yes | no | no | no | 5/21 | 0/8 |
| Transformers.js (`transformersjs`) | active | yes | no | yes | yes | 17/21 | 2/8 |
| Transformers.js v4 (webgpu) (`transformersjs-simulatte`) | experimental | no | yes | no | no | 14/21 | 0/8 |
| Transformers.js v4 (webgpu npm) (`transformersjs-webgpu-npm`) | experimental | no | yes | no | no | 14/21 | 0/8 |
| Transformers.js v4 (Bun WebGPU) (`transformersjs-bun`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Transformers.js v4 (Deno WebGPU) (`transformersjs-deno`) | experimental | no | no | no | no | 14/21 | 0/8 |
| Ratchet (`ratchet`) | experimental | yes | no | no | no | 5/21 | 0/8 |
| Candle (`candle`) | experimental | no | no | no | no | 4/21 | 0/8 |
| Burn (`burn`) | experimental | no | no | no | no | 4/21 | 0/8 |
| MediaPipe LLM Inference (`mediapipe-llm`) | active | yes | no | no | no | 5/21 | 0/8 |
| Wllama (`wllama`) | active | no | no | no | no | 4/21 | 0/8 |

## Model Coverage

| Doppler Model | In Catalog | Catalog Modes | TJS Mapping | Surface | Source | Compare Lane | Notes |
|---|---|---|---|---|---|---|---|
| `gemma-3-1b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-3-1b-it-ONNX-GQA` | auto | quickstart-registry | performance_comparable |  |
| `gemma-3-270m-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` | auto | quickstart-registry | performance_comparable |  |
| `gemma-4-e2b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-4-E2B-it-ONNX` | browser | local | performance_comparable |  |
| `gemma-4-e2b-it-q4k-ehf16-af32-int4ple` | yes | run, translate | `onnx-community/gemma-4-E2B-it-ONNX` | browser | local | performance_comparable | Doppler uses INT4 per-row PLE quantization (closer to TFLite shape); TJS uses standard ONNX q4f16. Both produce coherent Gemma 4 output on matching prompts — lane remains performance_comparable as compute-throughput comparison is meaningful. |
| `google-embeddinggemma-300m-q4k-ehf16-af32` | yes | embedding | `onnx-community/embeddinggemma-300m-ONNX` | auto | quickstart-registry | capability_only | Embedding models use a separate workload contract and are not part of the text-generation compare lane. |
| `qwen-3-5-0-8b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-0.8B-ONNX` | browser | quickstart-registry | performance_comparable |  |
| `qwen-3-5-2b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-2B-ONNX` | auto | quickstart-registry | capability_only | Qwen3.5 2B has the direct Transformers.js WebGPU runner path but is not yet promoted to a claimable compare lane. |
| `translategemma-4b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/translategemma-text-4b-it-ONNX` | auto | local | performance_comparable |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json) | Gemma 3 1B Instruct (Q4K/F32a) (996.4 MiB) | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-29 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/qwen3-5-0-8b-p064-d064-t0-k1.compare.json) | Qwen 3.5 0.8B (Q4K) (761.1 MiB) | 64 | 64 | greedy (t=0) | Apple M3; metal; darwin; chromium | 2026-03-30 |
| `p064-d064-t1-k32` | not captured | 64 | 64 | t=1, k=32, p=1 | not captured | not captured |
| `p256-d128-t0-k1` | not captured | 256 | 128 | greedy (t=0) | not captured | not captured |
| `p512-d128-t0-k1` | not captured | 512 | 128 | greedy (t=0) | not captured | not captured |
| `p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)
- [compare_gemma4_e2b_warm_cold_phases.svg](../benchmarks/vendors/results/compare_gemma4_e2b_warm_cold_phases.svg)

