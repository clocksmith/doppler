# Local GPU Challenger Framework

`benchmarks/vendors/local-gpu-challenger-matrix.json` is the canonical
multi-platform plan for fair local-GPU comparisons beyond the existing
Transformers.js anchor lanes.

The matrix keeps three things separate:

- Model target: the Doppler artifact and public model goal.
- Shared benchmark contract: workload, sampling, cache mode, correctness gates,
  sample statistics, and claim grade.
- Engine overlay: runner-specific backend, execution provider, format, delegate,
  and session settings.

## Current Tiers

| Tier | Model | Anchor | Local challengers |
| --- | --- | --- | --- |
| 0 | Gemma 3 270M | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; llama.cpp Vulkan GGUF |
| 0 | EmbeddingGemma 300M | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; HF Transformers PyTorch ROCm blocked until ROCm torch is installed |
| 0 | Gemma 3 1B | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; llama.cpp Vulkan GGUF |
| 1 | Qwen 3.5 0.8B | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; llama.cpp Vulkan GGUF |
| 1 | Qwen 3 Embedding 0.6B | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; HF Transformers PyTorch ROCm blocked until ROCm torch is installed |
| 1 | Qwen 3 Reranker 0.6B | Transformers.js WebGPU ONNX configured artifact, not a performance claim | HF Transformers PyTorch ROCm blocked until ROCm torch is installed; ONNX Runtime WebGPU direct |
| 2 | Qwen 3.5 2B | Transformers.js WebGPU ONNX | ONNX Runtime WebGPU direct; llama.cpp Vulkan GGUF |
| 2 | Gemma 4 E2B INT4 PLE | Transformers.js WebGPU ONNX | LiteRT GPU; HF Transformers PyTorch ROCm blocked until ROCm torch is installed |

Gemma 4 E2B is tracked as one model-level target. The selected Doppler artifact
for this framework is `gemma-4-e2b-it-q4k-ehf16-af16-int4ple`; the AF32 sibling
is listed only as an alternate artifact.

## Platform Scope

The matrix is intentionally not limited to the current Linux AMD workstation.
It tracks a multi-platform target set so future evidence can land without
rewriting the benchmark policy:

- Apple Metal
- Linux AMD Vulkan/ROCm
- Linux NVIDIA Vulkan/CUDA
- Linux Intel Vulkan
- Windows AMD WebGPU/DirectML
- Windows NVIDIA WebGPU/CUDA
- Windows Intel WebGPU
- NVIDIA Orin/Spark Linux

Each platform target must record its runtime surfaces separately from the shared
benchmark contract. A platform target being listed is not evidence that Doppler
wins there; it is only an allowed lane for fair, gated evidence.

## Current Probe Host

This host has AMD Strix Halo exposed through Vulkan/RADV and ROCm/HSA tooling.
Doppler WebGPU/Vulkan and Transformers.js WebGPU are usable local GPU stacks.
ONNX Runtime WebGPU direct is the next runner to add because the ONNX package is
installed and the Qwen ONNX artifacts exist.

PyTorch GPU is not currently usable for challenger claims on this host: the
installed torch build is CUDA, `torch.cuda.is_available()` is false, and
`torch.version.hip` is unset. HF Transformers PyTorch is therefore CPU-only for
diagnostics until a ROCm-enabled torch build is installed.

## Claim Gates

Every local challenger harness must expose:

- artifact identity
- format disclosure
- runtime surface
- hardware identity
- fallback status
- cache mode
- timing scope, including model load, compute phase, readback, total, and phase labels
- correctness before speed
- work accounting
- raw samples with p50/p95/p99
- claim grade

The valid claim grades are `diagnostic`, `local-gpu-comparable`, and
`release-claimable`. The matrix starts at `local-gpu-comparable`; release claims
still require hosted Doppler artifacts and pinned competitor revisions.

## Commands

```bash
node tools/local-gpu-challengers.js
node tools/local-gpu-challengers.js --check
node tools/local-gpu-challengers.js --json
node tools/local-gpu-challengers.js --probe-local --json
```

`--probe-local` reports host runner availability, including the Python torch
build and GPU backend state. The default check validates only checked-in
contracts, so CI does not depend on local GPU tools.
