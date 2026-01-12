# DOPPLER

**D**istributed **O**n-device **P**ipeline **P**rocessing **L**arge **E**mbedded **R**eploid

Browser-native WebGPU inference engine enabling tight CPU↔GPU co-evolution with [Reploid](https://github.com/clocksmith/reploid).

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

The converter embeds all model-specific inference parameters in `manifest.json`.
At runtime, overrides merge with the manifest, and the pipeline reads config
values directly (no model-family detection). Missing fields fail fast; `null`
explicitly disables a feature. Runtime overrides only apply when a value is
non-null; runtime `null` does not unset a manifest value.
Default kernel paths are resolved at conversion time via
`manifest.inference.defaultKernelPath`, with runtime overrides in
`runtime.inference.kernelPath`.
Runtime defaults to F16 activations for web inference; override with
`runtime.inference.compute.activationDtype = "f32"` if needed. Converter
manifests can also include `quantizationInfo.compute` as a hint.

## Quick Start

```bash
npm install
npm start         # Dev server at http://localhost:8080
npm run bench     # Run benchmarks
```

## Why Pure JS + WGSL

DOPPLER uses **JavaScript source code** with **hand-written WGSL kernels**. No TypeScript compilation, no TVM compiler, no WASM runtime.

**The math:** GPU compute is 96% of decode time. JS orchestration is 2%. Optimizing 2% with WASM doesn't matter.

| | WebLLM (TVM/WASM) | DOPPLER (JS/WGSL) |
|---|---|---|
| Unit of distribution | Compiled model binary | Weight shards + shared kernels |
| Runtime LoRA | Impossible (fused at compile) | Hot-swap at runtime |
| Expert paging | Fixed at compile | Dynamic (bind different buffers) |
| Device-specific kernels | One binary fits all | Per-device optimization |
| Debugging | Hard (compiled) | Chrome DevTools |

### Why JavaScript over TypeScript

DOPPLER source is JavaScript (not TypeScript) to enable **runtime hot-swap** of inference code without a build step.

**TypeScript compiles to JavaScript anyway.** The question is whether the compilation step adds value. For hot-swappable, agent-generated code, it adds friction without benefit.

**Agents will generate nearly 100% of this code**—if not now, very soon. No benchmark shows LLMs generate better TypeScript than JavaScript ([GitHub Octoverse 2025](https://github.blog/news-insights/octoverse/typescript-python-and-the-ai-feedback-loop-changing-software-development/)). Types help agents *read* context, not *write* better code ([Anders Hejlsberg](https://github.blog/developer-skills/programming-languages-and-frameworks/typescripts-rise-in-the-ai-era-insights-from-lead-architect-anders-hejlsberg/))—so every module has a `.d.ts` file that agents read directly.

| Concern | Resolution |
|---------|------------|
| **Type safety** | Tests catch type errors pre-production ([ICSE 2017](https://earlbarr.com/publications/typestudy.pdf)) |
| **Type specs for agents** | Every module has a `.d.ts` file; agents read these directly |
| **Consumer compatibility** | TypeScript users import with full type safety via `.d.ts` |
| **Hot-swap** | JS/WGSL/JSON swap at runtime; no recompilation |

See [Language Policy](docs/style/GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) for full rationale and citations.

## Model Support

| Architecture | Examples | Status |
|-------------|----------|--------|
| Gemma | Gemma 3 1B, 4B | Full support |
| LLaMA | LLaMA 2/3, Mistral | Full support |
| Mixtral | Mixtral 8x7B | MoE support |
| GPT-OSS | GPT-OSS 20B MoE | Experimental |

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for full documentation.

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
