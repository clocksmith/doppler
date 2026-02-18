# DOPPLER

**D**istributed **O**n-device **P**rocessing for **P**refill, **L**earning, and **E**xecution **R**untime

**[Try it live](https://d4da.com)**

Browser-native WebGPU runtime for forward inference, prefill, backward/training primitives, diffusion sampling, and energy-based inference.
Doppler is a standalone inference library; [Reploid](https://github.com/clocksmith/reploid) is an optional orchestrator integration.

## Performance vs Transformers.js (ORT WebGPU)

Gemma 3 1B F16, both models on local disk, headless Chrome, Apple M-series.
Reproducing: `node tools/compare-engines.mjs --tjs-version 4 --mode all --model-id gemma-3-1b-it --tjs-local-model-path /models/local/`

| Metric | Doppler | Transformers.js v4 | Delta |
|--------|---------|-------------------|-------|
| **Cold load (no cache)** | **3.1s** | 7.5s | 2.4x faster |
| **Warm load (OPFS cached)** | **3.2s** | 4.7s | 1.5x faster |
| Decode tok/s | 9.0 | 11.1 | 24% slower |
| Prefill tok/s | 49.4 | 90.7 | 84% slower |
| TTFT | 187ms | 99ms | 88% slower |

Doppler's RDRR format loads weights directly into GPU buffers with zero graph compilation.
For short-output use cases (intent classification, tool selection, autocomplete), total
time-to-first-useful-output is dominated by model load -- where Doppler is 2.4x faster cold.

Decode and prefill throughput gaps are active kernel optimization targets.

### Crossover Intuition (Rough)

- If total latency is approximated as `load + TTFT + decode`, Doppler's load advantage is erased around:
- Cold run: ~205 generated tokens
- Warm run: ~67 generated tokens
- Prefill being much slower for Doppler shifts crossover earlier for long prompts.

This favors product surfaces where outputs are intentionally short and frequent (intent head, tool routing, planner selection). In those browser-native flows, perceived UX and time-to-first-useful-output are often dominated by startup and load behavior, where Doppler leads.

### Benchmark Quality Notes

- Good: same model class, local disk, headless Chrome, reproducible command (`README.md:12-13`).

## Why This Works

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
npm run convert -- <inputDir> models/local/<id> --model-id <id>
npm run debug -- --model-id <id> --model-url /models/local/<id> --runtime-preset modes/debug
npm run bench -- --model-id <id> --model-url /models/local/<id> --runtime-preset experiments/gemma3-bench-q4k
npm run test:model -- --suite inference --model-id <id> --model-url /models/local/<id>
```

`bench` and `debug` can run via browser relay by default
(`--surface browser` / `--surface auto`), with Node-side execution as an optional
fallback.

Use `--headed` for headed mode (default is headless), or explicitly set
`--headless false`.

## Agent Setup

- `AGENTS.md` is canonical.
- `CLAUDE.md` and `GEMINI.md` are symlink aliases to `AGENTS.md`.
- Shared skills live in `skills/`, with provider aliases at `.claude/skills` and `.gemini/skills`.

Validate parity:

```bash
npm run agents:verify
```

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

Start at `docs/index.md`, then:
- `docs/architecture.md`
- `docs/config.md`
- `docs/formats.md`
- `docs/operations.md`
- `docs/testing.md`

## Requirements

- WebGPU browser (Chrome 113+, Edge 113+, Firefox Nightly)
- GPU with 4GB+ VRAM for 7B models

## Related

- [REPLOID](https://github.com/clocksmith/reploid) - Optional browser-native AI agent integration ([replo.id/r](https://replo.id/r))

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
