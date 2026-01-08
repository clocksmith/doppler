# Hardware Compatibility Matrix

Testing status for Gemma 3 1B and other DOPPLER models across different GPU/browser combinations.

## Tested Configurations

| GPU | Vendor | VRAM | Browser | OS | Status | Notes | Date Tested |
|-----|--------|------|---------|----|----|-------|-------------|
| Apple M3 | Apple | Unified 16GB+ | Chrome/Safari | macOS | ✓ WORKING | Gemma 3 1B inference validated | 2025-01 |
| AMD RX 395 | AMD | TBD | TBD | Linux | ⏳ TESTING | In progress | 2025-12-14 |

## Pending Tests

| GPU | Vendor | VRAM | Browser | OS | Priority | Target Model |
|-----|--------|------|---------|----|----|--------------|
| Apple M1 | Apple | Unified 8-16GB | Chrome/Safari | macOS | P1 | Gemma 3 1B |
| Apple M2 | Apple | Unified 8-24GB | Chrome/Safari | macOS | P1 | Gemma 3 1B |
| NVIDIA RTX 3080 | NVIDIA | 10GB | Chrome/Edge | Windows/Linux | P1 | Gemma 3 1B |
| NVIDIA RTX 4090 | NVIDIA | 24GB | Chrome/Edge | Windows/Linux | P1 | Larger models |
| AMD RX 6800 | AMD | 16GB | Chrome/Edge | Windows/Linux | P2 | Gemma 3 1B |
| Intel Arc A770 | Intel | 16GB | Chrome/Edge | Windows/Linux | P2 | Gemma 3 1B |

## Browser WebGPU Support

| Browser | Version | WebGPU Status | F16 Support | Subgroups | Notes |
|---------|---------|---------------|-------------|-----------|-------|
| Chrome | 113+ | ✓ Stable | ✓ | ✓ | Best compatibility; Android 121+ |
| Edge | 113+ | ✓ Stable | ✓ | ✓ | Chromium-based |
| Safari | 26+ | ✓ Stable | ✓ | ✓ | macOS Tahoe/iOS 26+; 17.4-18.x behind flag |
| Firefox | 141+ | ✓ Stable (Windows) | ✓ | ✓ | macOS 145+; Linux/Android behind flag |

**Sources:** [Can I Use WebGPU](https://caniuse.com/webgpu), [web.dev WebGPU announcement](https://web.dev/blog/webgpu-supported-major-browsers), [Mozilla GFX Blog](https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/)

## Known Issues by Platform

### Apple Silicon (M1/M2/M3)
- **Status**: Working
- **Strengths**: Unified memory allows larger models (no PCIe copy overhead)
- **Limitations**: None known
- **Recommended Browser**: Safari or Chrome

### AMD GPUs
- **Status**: Testing in progress
- **Strengths**: Good WebGPU support in recent drivers
- **Limitations**: TBD
- **Recommended Browser**: Chrome or Edge
- **Driver Requirements**: Mesa 23.0+ (Linux) or Adrenalin 23.0+ (Windows)

### NVIDIA GPUs
- **Status**: Untested
- **Strengths**: Best performance/watt, wide VRAM options
- **Limitations**: Discrete GPU (PCIe overhead for buffer transfers)
- **Recommended Browser**: Chrome or Edge
- **Driver Requirements**: 525+ (Linux) or Game Ready 525+ (Windows)

### Intel Arc GPUs
- **Status**: Untested
- **Strengths**: Good value, 16GB VRAM on A770
- **Limitations**: Newer architecture, driver maturity
- **Recommended Browser**: Chrome or Edge

## Model VRAM Requirements

| Model | Quantization | Minimum VRAM | Recommended VRAM | Notes |
|-------|--------------|--------------|------------------|-------|
| Gemma 3 1B | Q4_K_M | 1.2GB | 2GB | Includes KV cache overhead |
| Gemma 3 4B | Q4_K_M | 3.5GB | 6GB | — |
| GPT-OSS 20B | Q4_K_M | 12GB | 16GB | MoE model, 32 experts, 4 active per token |
| LLaMA 2 7B | Q4_K_M | 5GB | 8GB | — |
| Mistral 7B | Q4_K_M | 5GB | 8GB | — |
| Mixtral 8x7B | Q4_K_M | 28GB | 32GB | MoE model, all experts |

## Model Compatibility Status

### Tested End-to-End

| Model | Status | Platform | Notes | Date |
|-------|--------|----------|-------|------|
| Gemma 3 1B | ✓ WORKING | Mac M3 | Coherent output verified | 2025-01 |
| Gemma 3 1B | ⏳ TESTING | AMD Strix Halo (Linux) | Conversion complete, browser test pending | 2025-12-14 |
| GPT-OSS 20B MoE | ⏳ PARTIAL | Mac M3 | Router fixed (BF16), expert loading in progress | 2025-12-14 |

### Architecture Support (No E2E Testing Yet)

The following models have architecture support implemented but lack confirmed end-to-end test results:

| Model | Architecture | Quantization | Est. VRAM | Status |
|-------|--------------|--------------|-----------|--------|
| Gemma 3 4B | Dense transformer | Q4_K_M | 3.5GB | Untested |
| LLaMA 2 7B | Dense transformer | Q4_K_M | 5GB | Untested |
| LLaMA 3 7B | Dense transformer | Q4_K_M | 5GB | Untested |
| Mistral 7B | Dense transformer | Q4_K_M | 5GB | Untested |
| Mixtral 8x7B | MoE (8 experts) | Q4_K_M | 28GB | Untested |

**Note**: Architectural support exists (kernels, quantization, etc.) but these models need conversion and browser testing to confirm full compatibility.

## Test Checklist

When testing on new hardware, verify:

- [ ] WebGPU device detection succeeds
- [ ] Model loads without errors
- [ ] Prefill phase completes (prompt processing)
- [ ] Decode phase generates coherent tokens (not `<unused16>` garbage)
- [ ] No buffer allocation errors
- [ ] Performance is reasonable (>10 tok/s for 1B models)

### Expected Output

For prompt "the sky is", Gemma 3 1B should generate coherent continuations like:
- "blue"
- "cloudy"
- "filled with stars"

**BAD**: `<unused16>` or other special tokens indicate quantization/dequantization errors.

## How to Test

### 1. Run E2E Test (Automated)

```bash
# From reploid root
doppler test demo --headed           # Demo UI test
doppler test inference               # Quick inference validation
doppler bench inference --headed     # Full inference benchmark
```

### 2. Run Demo (Manual)

```bash
# From reploid root
npx serve .

# Open http://localhost:3000/doppler/
# Select Gemma 3 1B
# Wait for model load
# Try prompt: "the sky is"
```

### 3. Collect System Info

```bash
# GPU info (Linux)
lspci | grep VGA
glxinfo | grep "OpenGL renderer"

# Browser version
google-chrome --version

# WebGPU features (in browser console)
const adapter = await navigator.gpu.requestAdapter();
console.log(await adapter.requestAdapterInfo());
console.log(adapter.features);
```

## Contributing Test Results

To add your test results:

1. Test Gemma 3 1B following checklist above
2. Record:
   - GPU model and VRAM
   - Browser and version
   - OS and version
   - WebGPU features (F16, subgroups)
   - Performance (tok/s if available)
   - Any errors or issues
3. Submit PR updating this matrix

## Reference Links

- [WebGPU Feature Matrix](https://webgpureport.org/)
- [WebGPU Browser Support](https://caniuse.com/webgpu)
- [DOPPLER Architecture](./ARCHITECTURE.md)
- [Gemma 3 Debug Postmortem](./GEMMA3-DEBUG-POSTMORTEM.md)

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes, CLI flags (`--kernel-path`, `--kernel-profile`), and the OPFS purge helper.

