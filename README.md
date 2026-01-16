# DOPPLER

**D**istributed **O**n-device **P**ipeline **P**rocessing **L**arge **E**mbedded **R**eploid

Browser-native WebGPU inference engine enabling tight CPU↔GPU co-evolution with [Reploid](https://github.com/clocksmith/reploid).
Doppler is the engine; Reploid is the driver.

**[Try it live](https://replo.id/d)** | **[GitHub](https://github.com/clocksmith/doppler)**

## Why This Works

Doppler and Reploid share a browser process. Kernel updates apply without process restart.

| Capability | Claim |
|------------|-------|
| **80% native performance** | [WebLLM 2024](https://arxiv.org/abs/2412.15803) |
| **JIT kernel generation** | Hours → seconds ([nnJIT MobiSys 2024](https://dl.acm.org/doi/10.1145/3643832.3661892)) |
| **Kernel hot-swap** | Runtime shader creation ([W3C WGSL Spec](https://www.w3.org/TR/WGSL/)) |
| **Shared memory** | CPU↔GPU via SharedArrayBuffer ([WgPy 2025](https://arxiv.org/pdf/2503.00279), [WebGPU Explainer](https://gpuweb.github.io/gpuweb/explainer/)) |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Demo UI                          │
├─────────────────────────────────────────────────────┤
│             DOPPLER Inference Pipeline              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Tokenize │→│ Forward  │→│ Sample   │→ tokens    │
│  └──────────┘ └──────────┘ └──────────┘            │
├─────────────────────────────────────────────────────┤
│              GPU Kernels (WebGPU)                   │
│  MatMul │ RMSNorm │ RoPE │ Attention │ SiLU        │
├─────────────────────────────────────────────────────┤
│           Memory / Buffer Management                │
├─────────────────────────────────────────────────────┤
│  Storage (OPFS)  │  RDRR Loader  │  Tokenizer      │
└─────────────────────────────────────────────────────┘
```

## Manifest-First Config

The converter embeds model-specific inference parameters in `manifest.json`.
Runtime reads config directly (no model-family detection). Missing fields fail
fast; `null` explicitly disables a feature. Kernel paths resolve at conversion
time and can be overridden via `runtime.inference.kernelPath` or per-run context.
See `docs/CONFIG.md` and `docs/FORMATS.md` for the full contract.

## Quick Start

```bash
npm install
npm start         # Dev server at http://localhost:8080
npm run bench     # Run benchmarks
```

## Why Pure JS + WGSL

DOPPLER uses JavaScript orchestration with hand-written WGSL kernels so code can
hot-swap without a build step. GPU compute dominates decode time, so the focus
is on kernel performance and debuggability. Type contracts live in `.d.ts`
files; see `docs/style/GENERAL_STYLE_GUIDE.md` for the full rationale.

## Model Support

| Architecture | Examples | Status |
|-------------|----------|--------|
| Gemma | Gemma 3 1B, 4B | Full support |
| LLaMA | LLaMA 2/3, Mistral | Full support |
| Mixtral | Mixtral 8x7B | MoE support |
| GPT-OSS | GPT-OSS 20B MoE | Experimental |

## Documentation

Start at `docs/INDEX.md`, then:
- `docs/ARCHITECTURE.md`
- `docs/CONFIG.md`
- `docs/FORMATS.md`
- `docs/OPERATIONS.md`
- `docs/TESTING.md`
- `docs/ROADMAP.md`
- `docs/AGENT_INTENT_BUNDLE.md`

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
