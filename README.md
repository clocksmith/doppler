# DOPPLER

**D**istributed **O**n-device **P**ipeline **P**rocessing **L**arge **E**mbedded **R**eploid ([Reploid](https://github.com/clocksmith/reploid))

Browser-native WebGPU inference for decoding, diffusion sampling, and energy-based inference, plus a post-training engine for local model execution.
Doppler runs standalone as the engine; Reploid is an optional driver that can link in for orchestration.
In a neurosymbolic future, a combined stack is essential because decoding handles discrete reasoning, diffusion supplies rich generative priors, and energy models enforce constraints and verification across both.

**[Try it live](https://d4da.com)**

## Why This Works

Doppler and Reploid share a browser process. Kernel registry/config changes apply without rebuild; full kernel hot-swap is planned.

| Capability | Claim |
|------------|-------|
| **80% native performance** | [WebLLM 2024](https://arxiv.org/abs/2412.15803) |
| **JIT kernel generation** | Hours → seconds ([nnJIT MobiSys 2024](https://dl.acm.org/doi/10.1145/3643832.3661892)) |
| **Runtime WGSL compilation** | No build step for kernel changes ([W3C WGSL Spec](https://www.w3.org/TR/WGSL/)) |
| **Shared memory** | CPU↔GPU via SharedArrayBuffer ([WgPy 2025](https://arxiv.org/pdf/2503.00279), [WebGPU Explainer](https://gpuweb.github.io/gpuweb/explainer/)) |

## Quick Start

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080/demo/` for browser UI workflows.

Node CLI (shared command contract):

```bash
npm install --save-optional webgpu
npm run convert -- <inputDir> <outputDir> --model-id <id>
npm run debug -- --model-id <id> --runtime-preset modes/debug
npm run bench -- --model-id <id> --runtime-preset experiments/gemma3-bench-q4k
npm run test:model -- --suite inference --model-id <id>
```

`webgpu` is optional but recommended for Node CLI parity. Without it, harnessed
commands can still run via browser relay (`--surface browser` or `--surface auto` fallback).

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Browser App                          │
├────────────────────────────────────────────────────────────┤
│                 JS Runtime / Orchestrator                   │
│   Decode (LM) │ Diffusion (image/audio) │ Energy (EBM)      │
├────────────────────────────────────────────────────────────┤
│                  WGSL Kernel Pipeline                       │
│   MatMul │ Attention │ Conv │ Sampling │ Scoring            │
├────────────────────────────────────────────────────────────┤
│                       WebGPU Device                          │
├────────────────────────────────────────────────────────────┤
│  Memory/Buffer Mgmt │ Model Storage (OPFS) │ Tokenizer/IO    │
└────────────────────────────────────────────────────────────┘
```

## Manifest-First Config

The converter embeds model-specific inference parameters in `manifest.json`.
Runtime reads config directly (no model-family detection). Missing fields fail
fast; `null` explicitly disables a feature. Kernel paths resolve at conversion
time and can be overridden via `runtime.inference.kernelPath` or per-run context.
See `docs/config.md` and `docs/formats.md` for the full contract.

## Why Pure JS + WGSL

DOPPLER uses JavaScript orchestration with hand-written WGSL kernels so changes
compile at runtime without a build step (hot-swap plumbing is planned). GPU compute dominates decode time, so the focus
is on kernel performance and debuggability. Type contracts live in `.d.ts`
files; see `docs/style/general-style-guide.md` for the full rationale.

## Model Support

| Architecture | Examples | Status |
|-------------|----------|--------|
| Gemma | Gemma 3 1B, 4B | Full support |
| LLaMA | LLaMA 2/3, Mistral | Full support |
| Mixtral | Mixtral 8x7B | MoE support |
| GPT-OSS | GPT-OSS 20B MoE | Experimental |

## Documentation

Start at `docs/INDEX.md`, then:
- `docs/architecture.md`
- `docs/config.md`
- `docs/formats.md`
- `docs/operations.md`
- `docs/testing.md`

## Requirements

- WebGPU browser (Chrome 113+, Edge 113+, Firefox Nightly)
- GPU with 4GB+ VRAM for 7B models

## Related

- [REPLOID](https://github.com/clocksmith/reploid) - Browser-native AI agent ([replo.id/r](https://replo.id/r))

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

MIT
