# Release Matrix

Generated: 2026-03-10T12:52:59.866Z
Release: channel=main-snapshot, version=0.1.7, commit=065404704b57a7eafb16b5ef3144766c03173c3b, dirty=no

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
| `gemma-3-270m-it-q4k-ehf16-af32` | yes | run, translate |  |  | auto |  |
| `google-embeddinggemma-300m-q4k-ehf16-af32` | yes | embedding |  |  | auto |  |
| `translategemma-4b-it-q4k-ehf16-af32` | yes | run, translate |  |  | auto |  |

## Workloads

| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |
|---|---|---:|---:|---|---|---|
| `p064-d064-t0-k1` | not captured | 64 | 64 | greedy (t=0) | not captured | not captured |
| `p064-d064-t1-k32` | not captured | 64 | 64 | t=1, k=32, p=1 | not captured | not captured |
| `p256-d128-t0-k1` | not captured | 256 | 128 | greedy (t=0) | not captured | not captured |
| `p512-d128-t0-k1` | not captured | 512 | 128 | greedy (t=0) | not captured | not captured |
| `p256-d128-t1-k32` | not captured | 256 | 128 | t=1, k=32, p=1 | not captured | not captured |

## Charts

- [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

