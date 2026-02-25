# Doppler.js

**D**eterministic **O**n-device **P**rocessing for **P**refill, **L**oading, and **E**xecution **R**untime

**[Try it live](https://d4da.com)**

Doppler is a browser-native WebGPU runtime for local intent and inference loops.
It provides explicit load-path and kernel-path control, reproducible phase benchmarks against Transformers.js (v4), and auditable kernel execution tracing.

## Evidence

![Doppler vs Transformers.js v4 phase comparison (1B, selected warm OPFS workloads)](benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

This chart covers three 1B workloads (`g3-p032-d256-t0-k1`, `g3-p064-d064-t0-k1`, `g3-p064-d064-t1-k32`) under the same warm-opfs parity contract.
In this snapshot, Doppler leads first-response latency and decode throughput, while Transformers.js (v4) leads TTFT and prefill throughput.
Definitions and reproducibility details are in the Performance section.

## Local intent latency constraints

Intent pipelines are front-loaded: the first response determines perceived speed.
If load or TTFT drifts, the whole workflow feels delayed even when later decode is fast.
Browser-local inference keeps private prompt data local and makes control decisions resilient when connectivity is limited.
Doppler targets this critical path directly by making load mode, prefill, decode cadence, and kernel path explicit and tunable.

## Execution model: JavaScript + WGSL

Doppler keeps the runtime path thin and explicit.

Positioning:
- **vs WebLLM / TVM**: compiler-driven binaries can improve automation, but make shard-level composition, adapter hot-swap, and per-kernel inspection/tuning harder.
- **vs Transformers.js (v4) / ONNX Runtime**: Transformers.js (v4) runs through an ONNX graph/runtime layer; Doppler maps model + kernel path directly in config and exposes kernel-path tracing.

JS+WGSL advantages:
- **Control**: manifest-first contracts plus explicit kernel-path config keep execution choices auditable and reproducible.
- **Performance**: GPU compute dominates per-token latency, tensor ops stay on GPU, and readback is minimized to final logits.
- **Velocity**: handwritten WGSL kernels (human and/or coding agent) enable progressive fusion from debuggable atomic kernels to fused kernels without full model recompiles.

## Architecture comparison

```
Transformers.js                              Doppler
─────────────                                ───────
App (JS)                                     App (JS)
 └─ @huggingface/transformers (JS)            └─ Doppler runtime (JS)
     └─ onnxruntime-web (C++ → WASM)              └─ WebGPU
         └─ ONNX graph interpreter                     ├─ WGSL kernels (prewritten)
             └─ WebGPU                                 └─ RDRR weight shards
                 ├─ WGSL kernels (generated at
                 │   runtime from generic op library)
                 └─ ONNX weight shards
```

Transformers.js wraps a C++-to-WASM compiled ONNX runtime that interprets an exported graph and generates GPU shaders at runtime from a generic op library.
Models must be manually exported from PyTorch to ONNX format before they can run (maintained per-model by `onnx-community`).
Doppler is JS end-to-end with prewritten WGSL kernels per operation (attention, RoPE, RMSNorm, FFN) — no graph interpreter, no WASM layer, no runtime shader generation.
The RDRR format maps weight shards directly to GPU buffers, and conversion runs directly from SafeTensors or GGUF sources without an intermediate export step.
Architecture-specific patterns like sliding/full attention windows are runtime config in Doppler, not frozen in an exported graph.

## Runtime architecture and boundaries

![Doppler architecture overview](docs/architecture-overview.svg)

See [`docs/architecture.md`](docs/architecture.md) for full subsystem and boundary details.

## Current capabilities

- Unified browser/CLI command surface with reproducible cross-engine benchmark tooling.
- Manifest-driven model loading with explicit runtime overrides and kernel-path tracing.
- Local intent stack support (embedding retrieval + compact generative control outputs).
- In development: backward/training primitives, diffusion sampling, and energy-based inference.

## Performance

### Snapshot table (1B, warm-opfs, parity)

| Workload              | Engine               | Model load (ms) | TTFT (ms) | First response (ms) | Prefill (tok/s) | Decode (tok/s) |
| --------------------- | -------------------- | --------------: | --------: | ------------------: | --------------: | -------------: |
| `g3-p064-d064-t0-k1`  | Doppler              |      **3172.4** |     568.9 |          **3739.7** |          324.12 |      **23.40** |
| `g3-p064-d064-t0-k1`  | Transformers.js (v4) |          4738.6 | **300.8** |              5053.4 |      **608.34** |          21.06 |
| `g3-p064-d064-t1-k32` | Doppler              |      **3184.7** |     675.7 |          **3860.4** |          323.65 |      **23.26** |
| `g3-p064-d064-t1-k32` | Transformers.js (v4) |          4417.0 | **317.6** |              4734.5 |      **577.39** |          20.92 |

Based on repeated apples-to-apples runs (3 greedy + 2 sampling), Doppler is faster on first-response latency and decode throughput, while Transformers.js (v4) is faster on TTFT and prefill throughput.
Artifacts: `benchmarks/vendors/results/`.

### Reproduce

```bash
node tools/compare-engines.js --mode compute --cache-mode warm --load-mode opfs --decode-profile parity --tjs-version 4 --model-id gemma-3-1b-it-f16-f32a --tjs-model onnx-community/gemma-3-1b-it-ONNX-GQA --workload <workload-id> --warmup 1 --runs 1 --seed 0 --save --save-dir benchmarks/vendors/results
```

## Getting started

See [`docs/setup-instructions.md`](docs/setup-instructions.md) for install, conversion, and run guides.
For contribution workflow, see [`docs/contributing.md`](docs/contributing.md).
For disclosure and community policies, see [`SECURITY.md`](SECURITY.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Glossary

- **TTFT**: time-to-first-token from generation start.
- **First response**: `modelLoadMs + firstTokenMs`.
- **OPFS**: Origin Private File System browser storage used for persistent local model artifacts.
- **RDRR**: Doppler model artifact format used by loader/runtime contracts.
- **Parity profile**: one-token decode cadence used for cross-engine comparison.
- **Warm-opfs**: warm cache mode with OPFS local load path.

## Inspiration

- [WebLLM](https://github.com/mlc-ai/web-llm) - High-performance in-browser LLM inference
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [WebGPU](https://www.w3.org/TR/webgpu/) - W3C GPU API specification
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Sliding window attention, grouped-query attention
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - Sparse Mixture of Experts architecture
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) - Expert specialization in MoE
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) - Multi-head Latent Attention, 671B MoE
- [Kimi K2](https://arxiv.org/abs/2507.20534) - 1T parameter MoE, agentic intelligence
- [Dr. Doppler](https://megaman.fandom.com/wiki/Dr._Doppler) - Mega Man X3

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Trademarks

Trademark usage for the names "Doppler" and "Doppler.js" is described in
[BRANDING.md](BRANDING.md).
