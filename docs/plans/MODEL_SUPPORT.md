# Model Support Matrix

**Part of:** [VISION.md](../VISION.md) - Tracking model compatibility across phases

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **Working** | E2E inference verified |
| **Partial** | Loads but has issues |
| **Planned** | Architecture supported, not tested |
| **Blocked** | Missing kernel or feature |

---

## Hardware Compatibility

| Vendor | Device | WebGPU | Status | Notes |
|--------|--------|--------|--------|-------|
| **Apple** | M3 (Apple Silicon) | Metal | **Working** | Primary dev target, Gemma 3 1B verified |
| Apple | M1/M2 (Apple Silicon) | Metal | Untested | Should work (same Metal backend) |
| AMD | Strix Halo | Vulkan | Untested | Needs testing |
| AMD | RDNA 2/3 (desktop) | Vulkan | Untested | Needs testing |
| NVIDIA | RTX 30xx/40xx | Vulkan | Untested | Needs testing |
| Intel | Arc | Vulkan | Untested | Needs testing |
| Intel | Integrated (UHD) | Vulkan | Untested | May lack VRAM |

### Hardware-Specific Concerns

| Issue | Affected | Notes |
|-------|----------|-------|
| Unified memory | Apple Silicon | Advantage: GPU/CPU share RAM |
| VRAM limits | Discrete GPUs | Must fit model in dedicated VRAM |
| Vulkan vs Metal | Non-Apple | Different shader compilation paths |
| F16 support | Older GPUs | May need F32 fallback |
| Workgroup limits | All | WGSL kernels tuned for 256 threads |

### Testing New Hardware

```bash
# Check WebGPU adapter info
npx tsx tools/gpu-info.ts

# Run smoke test
npm run test:inference -- --headed
```

Report results by updating this matrix.

---

## Dense Models

| Model | Params | Quant | VRAM | Status | Tested On | Notes |
|-------|--------|-------|------|--------|-----------|-------|
| **Gemma 3 1B** | 1B | Q4_K_M | ~1GB | **Working** | Apple M3 | GELU activation, Q/K norm |
| Gemma 3 4B | 4B | Q4_K_M | ~3GB | Planned | - | Same arch as 1B |
| Llama 3.2 1B | 1B | Q4_K_M | ~1GB | Planned | - | Standard Llama |
| Llama 3.2 3B | 3B | Q4_K_M | ~2GB | Planned | - | Standard Llama |
| Llama 3.1 8B | 8B | Q4_K_M | ~5GB | Planned | - | Needs unified mem test |
| Mistral 7B | 7B | Q4_K_M | ~5GB | Planned | - | Standard Llama-like |

---

## MoE Models

| Model | Params | Active | Experts | Top-K | Quant | Status | Notes |
|-------|--------|--------|---------|-------|-------|--------|-------|
| **Phi-mini-MoE** | 7.6B | 2.4B | 16 | 2 | Q4_K_M | Planned | Smallest MoE, edge target |
| **Qwen3-30B-A3B** | 30.5B | 3.3B | 128 | 8 | Q4_K_M | Planned | Best quality/size ratio |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | Q4_K_M | Planned | Phase 2 target |
| **GPT-OSS 20B** | 20B | ~5B | varies | 2 | MXFP4 | Partial | Router works, experts loading |
| **Qwen3-235B-A22B** | 235B | 22B | 128 | 8 | Q4_K_M | Planned | Server-class MoE |
| **Kimi K2** | **1T** | **32B** | 384+1 | 8 | FP8 | Stretch | P2P distributed target, MLA attn |

---

## Architecture Support

| Feature | Status | Models Affected |
|---------|--------|-----------------|
| Standard attention | Working | All dense |
| GQA (Grouped Query) | Working | Llama 3, Gemma, Qwen3, Mixtral |
| MLA (Multi-head Latent) | Planned | Kimi K2 (DeepSeek-style) |
| Sliding window attention | Working | Mistral, Gemma 3, GPT-OSS |
| Q/K normalization | Working | Gemma 3, Qwen3 |
| RoPE standard (10K) | Working | Llama, Phi-mini-MoE |
| RoPE high theta (1M) | Working | Gemma 3, Qwen3, Mixtral |
| RoPE theta 50K + YARN | Planned | Kimi K2 |
| RMSNorm | Working | All |
| GELU activation | Working | Gemma 3 |
| SwiGLU/SiLU activation | Working | Llama, Qwen3, Mixtral, Kimi K2 |
| MoE routing (softmax+topk) | Working | Mixtral, GPT-OSS, Qwen3 |
| MoE scatter-add | Working | Mixtral, GPT-OSS |
| MoE shared experts | Planned | Kimi K2 (1 shared) |
| Q4_K_M dequant | Working | Most models |
| MXFP4 dequant | Working | GPT-OSS |
| FP8 dequant | Planned | Kimi K2 |
| BF16 embeddings | Working | All |

---

## Quantization Support

| Format | Status | Notes |
|--------|--------|-------|
| **Q4_K_M** | Working | llama.cpp compatible |
| **MXFP4** | Working | GPT-OSS mixed-precision |
| **BF16** | Working | Embeddings, norms |
| Q8_0 | Planned | Higher precision option |
| F16 | Working | Intermediate activations |
| F32 | Working | Fallback |

---

## VRAM Requirements

| Model | Weights | KV Cache (4K ctx) | Total |
|-------|---------|-------------------|-------|
| Gemma 3 1B | ~0.8GB | ~0.1GB | ~1GB |
| Gemma 3 4B | ~2.5GB | ~0.3GB | ~3GB |
| Phi-mini-MoE (2 experts) | ~2GB | ~0.2GB | ~2.5GB |
| Llama 3.2 3B | ~1.8GB | ~0.2GB | ~2GB |
| Llama 3.1 8B | ~4.5GB | ~0.5GB | ~5GB |
| Mistral 7B | ~4GB | ~0.5GB | ~5GB |
| Qwen3-30B-A3B (8 experts) | ~8GB | ~0.3GB | ~9GB |
| Mixtral 8x7B (all experts) | ~26GB | ~0.5GB | ~27GB |
| Mixtral 8x7B (2 experts) | ~6GB | ~0.5GB | ~7GB |
| GPT-OSS 20B (4 experts) | ~8GB | ~0.3GB | ~8GB |
| Qwen3-235B-A22B (8 experts) | ~50GB | ~1GB | ~51GB |
| Kimi K2 (8 experts) | ~80GB | ~2GB | ~82GB |

---

## Test Commands

```bash
# Run inference test (default model)
npm run test:inference -- --headed

# Run with specific model
npm run test:inference -- --model gemma3-1b-q4 --headed

# Run with verbose output
npm run test:inference -- --headed --verbose
```

---

## Known Issues

| Model | Issue | Status | Workaround |
|-------|-------|--------|------------|
| ~~Gemma 3 1B~~ | ~~Wrong activation (SiLU→GELU)~~ | **Fixed** | ~~N/A~~ |
| GPT-OSS 20B | Expert loading slow | Open | Needs paging |
| All large | VRAM overflow | Open | Phase 3 (tiered memory) |
| All | Non-Apple untested | Open | Need community testing |

### Hardware Testing Priority

1. **AMD Strix Halo** - Important for gaming laptops with strong iGPU
2. **NVIDIA RTX 40xx** - Most common discrete GPU
3. **AMD RDNA 3** - Desktop GPU alternative
4. **Intel Arc** - Growing market share

---

## Conversion

Models are converted to RDRR format using:

```bash
npx tsx tools/convert-cli.ts \
  --input ~/models/model-name/ \
  --output ./models/model-name-q4/ \
  --quantize q4_k_m
```

See [RDRR_FORMAT.md](../spec/RDRR_FORMAT.md) for format details.

---

## Adding New Models

1. Check architecture support table above
2. Convert to RDRR with `convert-cli.ts`
3. Run inference test: `npm run test:inference -- --model <model-name> --headed`
4. If issues, check [DOPPLER-TROUBLESHOOTING.md](../DOPPLER-TROUBLESHOOTING.md)
5. Update this matrix with results

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

