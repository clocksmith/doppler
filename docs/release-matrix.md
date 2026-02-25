# Release Matrix

Generated: 2026-02-25T16:31:55.189Z
Release: channel=main-snapshot, version=n/a, commit=6080b5a22b6ba32c2df82c1917756f10c2db75db, dirty=yes

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

| Workload ID | Workload Name | Prefill | Decode | Sampling | GPU/OS/Platform | JSON Runs |
|---|---|---:|---:|---|---|---|
| `decode-64-128-greedy` | Decode 64/128 Greedy | 64 | 128 | t=0, k=1, p=1 | not captured | not captured |
| `decode-32-64-greedy` | Decode 32/64 Greedy | 32 | 64 | t=0, k=1, p=1 | not captured | not captured |
| `decode-512-128-greedy` | Decode 512/128 Greedy | 512 | 128 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p032-d016-t0-k1` | Gemma3 Grid 32/16 Greedy | 32 | 16 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p032-d064-t0-k1` | Gemma3 Grid 32/64 Greedy | 32 | 64 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p032-d256-t0-k1` | Gemma3 Grid 32/256 Greedy | 32 | 256 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p064-d016-t0-k1` | Gemma3 Grid 64/16 Greedy | 64 | 16 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p064-d064-t0-k1` | Gemma3 Grid 64/64 Greedy | 64 | 64 | t=0, k=1, p=1 | GPU: Apple / gpu-family-apple-9 / M3 Pro; Backend: metal; OS: darwin; Platform: MacIntel | [sample-compare.json](../benchmarks/vendors/fixtures/sample-compare.json) (2026-02-25) |
| `g3-p064-d064-t1-k32` | Gemma3 Grid 64/64 t1 k32 (sampling) | 64 | 64 | t=1, k=32, p=1 | not captured | not captured |
| `g3-p064-d256-t0-k1` | Gemma3 Grid 64/256 Greedy | 64 | 256 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p512-d016-t0-k1` | Gemma3 Grid 512/16 Greedy | 512 | 16 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p512-d064-t0-k1` | Gemma3 Grid 512/64 Greedy | 512 | 64 | t=0, k=1, p=1 | not captured | not captured |
| `g3-p512-d256-t0-k1` | Gemma3 Grid 512/256 Greedy | 512 | 256 | t=0, k=1, p=1 | not captured | not captured |
Captured workloads: 1/13

## Evidence

- Committed charts: 1
  - [compare_1b_multi-workload_favorable_phases.svg](../benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)
- Compare JSON artifacts: 1
  - [sample-compare.json](../benchmarks/vendors/fixtures/sample-compare.json) (workload `g3-p064-d064-t0-k1`, models `gemma-3-1b-it-f16-f32a` vs `onnx-community/gemma-3-1b-it-ONNX-GQA`, runtime `GPU: Apple / gpu-family-apple-9 / M3 Pro; Backend: metal; OS: darwin; Platform: MacIntel`) **(latest)**
- Selected latest compare: [sample-compare.json](../benchmarks/vendors/fixtures/sample-compare.json) (section `compute/parity`, models `gemma-3-1b-it-f16-f32a` vs `onnx-community/gemma-3-1b-it-ONNX-GQA`)

## Runtime Specs (Latest Compare)

| Field | Value |
|---|---|
| Host platform | darwin |
| Host arch | arm64 |
| Node runtime | v22.12.0 |
| Browser platform | MacIntel |
| Browser language | en-US |
| Browser vendor | Google Inc. |
| Browser executable | chromium |
| GPU adapter | Apple / gpu-family-apple-9 / M3 Pro |
| GPU backend | metal |
| GPU features | f16=yes, subgroups=yes, timestamp_query=yes |

