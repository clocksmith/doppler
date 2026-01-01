# DOPPLER

**D**istributed **O**n-device **P**ipeline **P**rocessing **L**arge **E**mbedded **R**eploid

Browser-native LLM inference engine powered by WebGPU.

[![npm](https://img.shields.io/npm/v/@clocksmith/doppler)](https://www.npmjs.com/package/@clocksmith/doppler)
[![GitHub](https://img.shields.io/github/license/clocksmith/doppler)](https://github.com/clocksmith/doppler/blob/main/LICENSE)

**[Try it live](https://doppler.dev)** | **[GitHub](https://github.com/clocksmith/doppler)**

## Features

- **WebGPU acceleration** - Custom WGSL kernels for attention, FFN, RMSNorm
- **Quantized models** - Q4_K_M and MXFP4 for efficient VRAM usage
- **Streaming inference** - Token-by-token generation with KV cache
- **RDRR format** - Sharded weights, on-demand loading from OPFS or remote
- **MoE support** - GPU-native expert routing with lazy expert loading

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

## Why Pure JS + WGSL

DOPPLER uses **JavaScript orchestration** with **hand-written WGSL kernels**. No TVM compiler, no WASM runtime.

**The math:** GPU compute is 96% of decode time. JS orchestration is 2%. Optimizing 2% with WASM doesn't matter.

| | WebLLM (TVM/WASM) | DOPPLER (JS/WGSL) |
|---|---|---|
| Unit of distribution | Compiled model binary | Weight shards + shared kernels |
| Runtime LoRA | Impossible (fused at compile) | Hot-swap at runtime |
| Expert paging | Fixed at compile | Dynamic (bind different buffers) |
| Device-specific kernels | One binary fits all | Per-device optimization |
| P2P integration | Awkward (binary blob) | Native (JS fetch/WebRTC) |
| Debugging | Hard (compiled) | Chrome DevTools |

**Unique capabilities enabled by this architecture:**
- Flash Attention in pure WGSL (no other browser framework has this)
- GPU-native MoE routing with custom scatter-add kernels
- Runtime kernel hot-swap for device-specific optimization
- Native Bridge for mmap access to local files (bypasses OPFS limits)

See [Competitive Analysis](docs/analysis/COMPETITIVE.md) for full technical comparison.

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm start           # Dev server at http://localhost:8080

# Run benchmarks
npm run bench:inference -- --headed
```

## Installation

```bash
npm install @clocksmith/doppler
```

```typescript
import { DopplerProvider } from '@clocksmith/doppler/provider';

// Initialize
await DopplerProvider.init();

// Load model
await DopplerProvider.loadModel('gemma-3-1b-q4k', modelUrl);

// Generate
for await (const token of DopplerProvider.stream(messages, config)) {
  console.log(token);
}
```

## Performance Tuning (Kernel Overrides)

DOPPLER can force specific kernel modes at runtime without rebuilding models. These overrides win over the manifest and are ideal for quick A/B tests.

```bash
# Force Q4K fused matmul (4-bit kernel mode) + F16 compute
doppler bench inference --model gemma-1b-q4-row \
  --q4k-matmul fused_q4k --compute-precision f16

# Force dequant path (baseline)
doppler bench inference --model gemma-1b-q4-row \
  --q4k-matmul dequant_f16 --compute-precision f16
```

Notes:
- Fused Q4K requires `q4kLayout=row_wise` and WebGPU subgroups.
- Column-wise Q4K layout will always fall back to dequant for correctness.

### Update Manifest (No Shard Changes)

Use this to persist kernel hints in the manifest without touching shards:

```bash
npx tsx doppler/tools/update-manifest.ts ./models/gemma-1b-q4-row \
  --q4k-matmul fused_q4k --compute-precision f16
```

Unsafe edits (layout changes) require `--allow-unsafe`.

## Model Support

| Architecture | Examples | Status |
|-------------|----------|--------|
| Gemma | Gemma 3 1B, 4B | Full support |
| LLaMA | LLaMA 2/3, Mistral | Full support |
| Mixtral | Mixtral 8x7B | MoE support |
| GPT-OSS | GPT-OSS 20B MoE | Experimental |

## P2P Evolution (Planned)

Weight shards use CDN (HuggingFace). P2P is for **dynamic components** that benefit from decentralized evolution:

| Component | Size | P2P Value |
|-----------|------|-----------|
| **LoRA adapters** | 50-200MB | Fine-tuned personalities, domain experts |
| **Router weights** | ~1MB | Learned MoE routing, hierarchical gating |
| **WGSL kernels** | ~5KB each | Device-specific optimizations |
| **Sampling strategies** | ~10KB | Novel decoding algorithms |

```
┌─────────────────────────────────────────────────────┐
│                  DOPPLER Swarm                      │
├─────────────────────────────────────────────────────┤
│  Peer A              Peer B              Peer C     │
│  ├─ LoRA: writer    ├─ LoRA: coder      ├─ LoRA: ? │
│  ├─ Router v2       ├─ Router v3        │          │
│  └─ Kernel: M3 Max  └─ Kernel: RTX 4090 │          │
│                                                     │
│  ◄──── LoRA/kernel/router exchange ────►           │
│  └────── swarm gossip: who has what ──────┘        │
└─────────────────────────────────────────────────────┘
```

See [Memory Tiers](docs/internals/MEMORY_TIERS.md) and [Competitive Analysis](docs/analysis/COMPETITIVE.md#p2p-and-evolution-potential) for details.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [Inference Pipeline](inference/README.md) - Kernel graphs and execution flow
- [RDRR Format](docs/spec/RDRR_FORMAT.md) - Model packaging specification
- [Competitive Analysis](docs/analysis/COMPETITIVE.md) - Landscape and differentiators

## Requirements

- WebGPU browser (Chrome 113+, Edge 113+, Firefox Nightly)
- GPU with 4GB+ VRAM for 7B models

## Related

- [REPLOID](https://github.com/clocksmith/reploid) - Browser-native AI agent with recursive self-improvement

## License

MIT

