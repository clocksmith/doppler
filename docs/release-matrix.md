# Release Matrix

Generated: 2026-07-04T00:47:41.651Z
Release: channel=main-snapshot, version=0.4.6, commit=5a1a83578f3989f12c0cbc90b3ca0e3d34e6acd5, dirty=yes

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
| `gemma-3-270m-it-f16-af32` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` | auto | local | performance_comparable |  |
| `gemma-3-270m-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-3-270m-it-ONNX` | auto | quickstart-registry | performance_comparable |  |
| `gemma-4-12b-it-text-q4k-ehf16-af16` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-12b-it-text-q4k-ehf16-af32` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-12b-it-text-w4a16-ct-ehf16-af16` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-31b-it-text-q4k-ehf16-af16` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-31b-it-text-q4k-ehf16-af32` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-e2b-it-q4k-ehf16-af16-int4ple` | yes | run, translate |  | auto |  |  |  |
| `gemma-4-e2b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/gemma-4-E2B-it-ONNX` | browser | local | performance_comparable | Doppler and the paired Transformers.js ONNX q4f16 runner both produce coherent Gemma 4 output, but current greedy text is not exact-match; this lane is claimable for compute-throughput comparisons, not correctness-parity claims. |
| `gemma-4-e2b-it-q4k-ehf16-af32-int4ple` | yes | run, translate | `onnx-community/gemma-4-E2B-it-ONNX` | browser | local | performance_comparable | Doppler uses INT4 per-row PLE quantization (closer to TFLite shape); TJS uses standard ONNX q4f16. Both produce coherent Gemma 4 output on matching prompts — lane remains performance_comparable as compute-throughput comparison is meaningful. |
| `google-embeddinggemma-300m-q4k-ehf16-af32` | yes | embedding | `onnx-community/embeddinggemma-300m-ONNX` | auto | quickstart-registry | capability_only | Embedding models use a separate workload contract and are not part of the text-generation compare lane. |
| `qwen-3-5-0-8b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-0.8B-ONNX` | browser | local | performance_comparable |  |
| `qwen-3-5-2b-q4k-ehaf16` | yes | run, translate | `onnx-community/Qwen3.5-2B-ONNX` | browser | local | capability_only | Qwen 3.5 2B has no committed correctness-clean fixture for the claimable compare lane yet. |
| `qwen-3-6-27b-q4k-eaf16` | yes | run, translate |  | auto |  |  |  |
| `qwen-3-6-27b-q4k-ehaf16` | yes | run, translate |  | auto |  |  |  |
| `qwen-3-embedding-0-6b-q4k-ehf16-af32` | yes | embedding |  | auto |  |  |  |
| `qwen-3-reranker-0-6b-f16-af32` | yes | run, translate |  | auto |  |  |  |
| `translategemma-4b-it-q4k-ehf16-af32` | yes | run, translate | `onnx-community/translategemma-text-4b-it-ONNX` | auto | local | performance_comparable |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Correctness | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|---|
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/g3-1b-p064-d064-t0-k1.compare.json) | Gemma 3 1B Instruct (Q4K/F32a) (996.4 MiB) | 64 | 64 | greedy (t=0) | exact | Apple M3; metal; darwin; chromium; doppler browser | 2026-03-29 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-p064-d064-t0-k1-strix-halo-20260627.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 64 | 64 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler browser | 2026-06-27 |
| [`p064-d064-t0-k1`](../benchmarks/vendors/fixtures/qwen3-5-0-8b-p064-d064-t0-k1.compare.json) | Qwen 3.5 0.8B (Q4K) (1.08 GiB) | 64 | 64 | greedy (t=0) | exact | Apple M3; metal; darwin; chromium; doppler browser | 2026-03-30 |
| `p064-d064-t1-k32` | not captured | 64 | 64 | t=1, k=32, p=1 | not captured | not captured | not captured |
| [`p256-d128-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-p256-d128-t0-k1-strix-halo-20260626.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 256 | 128 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler browser | 2026-06-27 |
| [`p512-d128-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-local-p512-d128-t0-k1-strix-halo-20260627.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 512 | 128 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler browser | 2026-06-27 |
| [`p512-d128-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 512 | 128 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler bun / command node | 2026-06-27 |
| [`p512-d128-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 512 | 128 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler node | 2026-06-27 |
| [`p512-d128-t0-k1`](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-p512-d128-t0-k1-strix-halo-20260626.compare.json) | Gemma 3 270M Instruct (Q4K/F32a) (399.1 MiB) | 512 | 128 | greedy (t=0) | exact | amd 0x1586; vulkan; linux; chromium; doppler browser | 2026-06-27 |
| `p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured | not captured |

## Local Claim Lanes

| Lane | Status | Gate gaps | Backend | Surface | Workload | Decode tok/s (Doppler/TJS) | Prompt tok/s (Doppler/TJS) | Leaders | Bottleneck | Evidence |
|---|---|---|---|---|---|---:|---:|---|---|---|
| `gemma-3-1b-it-q4k-rdrr (Gemma 3 1B Instruct (Q4K/F32a))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | bun-webgpu | bun | p064-d064-t0-k1 | 115.5 tok/s / 110.4 tok/s | 968.4 tok/s / 832.5 tok/s | decode Doppler; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T202736.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | bun-webgpu | bun | p256-d128-t0-k1 | 108.4 tok/s / 109.9 tok/s | 1969.8 tok/s / 992.2 tok/s | decode TJS; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T203220.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | bun-webgpu | bun | p512-d128-t0-k1 | 105.5 tok/s / 97.66 tok/s | 2323.7 tok/s / 1004 tok/s | decode Doppler; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T200603.json) / [svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627T200603.svg) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | chromium-webgpu | browser | p064-d064-t0-k1 | 175.8 tok/s / 107.1 tok/s | 1036.5 tok/s / 840.4 tok/s | decode Doppler; prompt Doppler | readback map wait (submit-readback-wait) | [compare](../benchmarks/vendors/results/compare_20260627T202837.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | chromium-webgpu | browser | p256-d128-t0-k1 | 164.2 tok/s / 113 tok/s | 2071.2 tok/s / 1021.5 tok/s | decode Doppler; prompt Doppler | readback map wait (submit-readback-wait) | [compare](../benchmarks/vendors/results/compare_20260627T203348.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | chromium-webgpu | browser | p512-d128-t0-k1 | 156.5 tok/s / 96.08 tok/s | 2426.6 tok/s / 1024 tok/s | decode Doppler; prompt Doppler | readback map wait (submit-readback-wait) | [compare](../benchmarks/vendors/results/compare_20260627T200811.json) / [svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | node-webgpu | node | p064-d064-t0-k1 | 112.5 tok/s / 108.2 tok/s | 1106 tok/s / 831.9 tok/s | decode Doppler; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T202549.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | node-webgpu | node | p256-d128-t0-k1 | 106 tok/s / 109.2 tok/s | 2156.3 tok/s / 1008.4 tok/s | decode TJS; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T203031.json) |
| `gemma-3-270m-it-q4k-rdrr (Gemma 3 270M Instruct (Q4K/F32a))` | candidate | status candidate | node-webgpu | node | p512-d128-t0-k1 | 103 tok/s / 97.15 tok/s | 2453.3 tok/s / 1021.5 tok/s | decode Doppler; prompt Doppler | command recording (command-recording) | [compare](../benchmarks/vendors/results/compare_20260627T200323.json) / [svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627T200323.svg) |
| `gemma-4-e2b-it-int4ple-rdrr (Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |
| `gemma-4-e2b-it-q4k-rdrr (Gemma 4 E2B Instruct (Q4K/F32a))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |
| `qwen-3-5-0-8b-q4k-rdrr (Qwen 3.5 0.8B (Q4K))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |
| `qwen-3-5-2b-q4k-rdrr (Qwen 3.5 2B (Q4K))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |
| `translategemma-4b-it-q4k-rdrr (TranslateGemma 4B Instruct (Q4K))` | candidate | status candidate; missing backends chromium-webgpu; missing workloads p064-d064-t0-k1, p256-d128-t0-k1, p512-d128-t0-k1; missing decode profiles parity, throughput; missing backend/workload chromium-webgpu:p064-d064-t0-k1, chromium-webgpu:p256-d128-t0-k1, chromium-webgpu:p512-d128-t0-k1 | not captured | not captured | not captured |  |  |  |  | missing |

## Latest Bottlenecks

Source: [gemma-3-270m-it-q4k-rdrr-local-p512-d128-t0-k1-strix-halo-20260627.compare.json](../benchmarks/vendors/fixtures/gemma-3-270m-it-q4k-rdrr-local-p512-d128-t0-k1-strix-halo-20260627.compare.json)

Doppler internal: readback map wait 530.9 ms; 63.4% of decode; command recording 302.5 ms; 19177 ops / 2 passes

| Metric | Leader | Gap | Doppler | Transformers.js |
|---|---|---:|---:|---:|
| model load | transformersjs | 81.7% | 1106.1 ms | 608.8 ms |
| first response (first token + load) | transformersjs | 53.97% | 1317.1 ms | 855.4 ms |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)
- [compare_gemma4_e2b_warm_cold_phases.svg](../benchmarks/vendors/results/compare_gemma4_e2b_warm_cold_phases.svg)
- [doppler-backend-evidence-summary.svg](../benchmarks/vendors/results/doppler-backend-evidence-summary.svg)
- [doppler-vulkan-decode-grid-20260627.svg](../benchmarks/vendors/results/doppler-vulkan-decode-grid-20260627.svg)
- [doppler-vulkan-p512-surface-sweep-20260627.svg](../benchmarks/vendors/results/doppler-vulkan-p512-surface-sweep-20260627.svg)
- [gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-browser-p512-d128-t0-k1-strix-halo-20260627T200811.svg)
- [gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627.svg)
- [gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627T200603.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-bun-p512-d128-t0-k1-strix-halo-20260627T200603.svg)
- [gemma-3-270m-it-q4k-rdrr-local-p512-d128-t0-k1-strix-halo-20260627.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-local-p512-d128-t0-k1-strix-halo-20260627.svg)
- [gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627.svg)
- [gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627T200323.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-node-p512-d128-t0-k1-strix-halo-20260627T200323.svg)
- [gemma-3-270m-it-q4k-rdrr-p064-d064-t0-k1-strix-halo-20260626.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-p064-d064-t0-k1-strix-halo-20260626.svg)
- [gemma-3-270m-it-q4k-rdrr-p256-d128-t0-k1-strix-halo-20260626.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-p256-d128-t0-k1-strix-halo-20260626.svg)
- [gemma-3-270m-it-q4k-rdrr-p512-d128-t0-k1-strix-halo-20260626.svg](../benchmarks/vendors/results/gemma-3-270m-it-q4k-rdrr-p512-d128-t0-k1-strix-halo-20260626.svg)
