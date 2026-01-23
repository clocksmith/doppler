# Doppler GPU Requirements & Performance Metrics

This document describes GPU hardware requirements for running Doppler WebGPU inference and expected performance metrics.

## WebGPU Tier System

Doppler automatically detects GPU capabilities and assigns a tier level that determines which models can run efficiently.

### Tier 1: Unified Memory (Best)

**Hardware Examples:**
- Apple Silicon (M1/M2/M3/M4) Mac with 16GB+ unified memory
- Snapdragon X Elite laptops
- Future AMD APUs with large unified memory

**Capabilities:**
- `memory64`: Large buffer support (8GB+)
- `subgroups`: Optimized reduction operations
- `shader-f16`: Native FP16 compute
- Unified memory: No CPU-GPU transfer overhead

**Max Model Size:** ~60GB (with swapping)

**Recommended Models:**
- Gemma 3 12B Q4_K_M
- Gemma 3 4B Q4_K_M
- Gemma 3 1B Q4_K_M
- Mixtral 8x7B (MoE)

### Tier 2: Memory64

**Hardware Examples:**
- NVIDIA RTX 3090/4090 (24GB VRAM)
- NVIDIA RTX 4080 (16GB VRAM)
- AMD RX 7900 XTX (24GB VRAM)

**Capabilities:**
- `memory64`: Large buffer support (2-8GB)
- `subgroups`: Optimized reduction operations
- Discrete GPU with dedicated VRAM

**Max Model Size:** ~40GB (MoE models with expert offloading)

**Recommended Models:**
- Gemma 3 4B Q4_K_M
- Gemma 3 1B Q4_K_M
- Phi-3 Mini

### Tier 3: Basic

**Hardware Examples:**
- Intel Integrated Graphics (UHD 620+)
- AMD Integrated Graphics (Vega 8+)
- NVIDIA GTX 1060/1070
- Entry-level laptops

**Capabilities:**
- Basic WebGPU support
- Limited buffer sizes (<2GB)
- May lack some optimizations

**Max Model Size:** ~8GB

**Recommended Models:**
- Gemma 3 1B Q4_K_M (primary recommendation)
- SmolLM 135M (fallback)

## Model VRAM Requirements

| Model | Params | Quant | VRAM Required | Min Tier | Notes |
|-------|--------|-------|---------------|----------|-------|
| Gemma 3 1B Q4_K_M | 1B | Q4_K_M | 1.2 GB | 3 | Works on integrated GPUs |
| Gemma 3 4B Q4_K_M | 4B | Q4_K_M | 2.8 GB | 2 | Requires discrete GPU |
| Gemma 3 12B Q4_K_M | 12B | Q4_K_M | 7.5 GB | 1 | Requires 8GB+ VRAM |
| Gemma 3 27B Q4_K_M | 27B | Q4_K_M | 16 GB | 1 | Apple Silicon or RTX 4090 |
| Phi-3 Mini | 3.8B | Q4_K_M | 2.4 GB | 2 | Good for coding tasks |
| Mixtral 8x7B | 47B | Q4_K_M | 28 GB | 1 | MoE, needs expert offload |

### Memory Formula

Approximate VRAM calculation:

```
VRAM = (params_in_billions * bits_per_weight / 8) + kv_cache + activations

For Q4_K_M (~4.5 bits avg):
VRAM_GB = params_B * 0.56 + 0.5 (overhead)

For Q8:
VRAM_GB = params_B * 1.0 + 0.5

For F16:
VRAM_GB = params_B * 2.0 + 0.5
```

## Performance Expectations

### Token Generation Speed (tok/s)

| Model | Tier 1 (M3 Max) | Tier 2 (RTX 4090) | Tier 3 (Intel UHD) |
|-------|-----------------|-------------------|---------------------|
| Gemma 3 1B Q4 | 80-120 | 100-150 | 15-30 |
| Gemma 3 4B Q4 | 40-60 | 60-80 | N/A |
| Gemma 3 12B Q4 | 15-25 | 25-35 | N/A |

### Time-to-First-Token (TTFT)

| Model | Tier 1 | Tier 2 | Tier 3 |
|-------|--------|--------|--------|
| Gemma 3 1B Q4 | 200-400ms | 150-300ms | 500-1000ms |
| Gemma 3 4B Q4 | 400-800ms | 300-500ms | N/A |
| Gemma 3 12B Q4 | 1-2s | 800ms-1.5s | N/A |

## LoRA Adapter Overhead

LoRA adapters add minimal overhead to inference:

| Rank | Additional VRAM | Speed Impact |
|------|-----------------|--------------|
| 8 | ~10-20 MB | <2% |
| 16 | ~20-40 MB | <3% |
| 32 | ~40-80 MB | <5% |
| 64 | ~80-160 MB | <8% |

### Adapter Switching

- **Cold switch:** ~100-500ms (loading new weights)
- **Hot switch (cached):** <50ms (pre-loaded in memory)
- **Adapter composition:** +10-20% inference time per additional adapter

## Browser Requirements

### Supported Browsers

| Browser | WebGPU Status | Notes |
|---------|---------------|-------|
| Chrome 113+ | Full support | Recommended |
| Chrome Canary | Full support | Latest features |
| Edge 113+ | Full support | Chromium-based |
| Safari 18+ | Full support | Best on Apple Silicon |
| Firefox Nightly | Experimental | Behind `dom.webgpu.enabled` |

### Required WebGPU Features

**Essential:**
- `GPUAdapter.requestDevice()`
- Storage buffers (compute shaders)
- Timestamp queries (for profiling)

**Recommended:**
- `shader-f16` (FP16 compute)
- `subgroups` (faster reductions)

### Memory Limits

WebGPU has browser-enforced limits that affect model loading:

```javascript
// Check limits in browser console:
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log('Max buffer size:', device.limits.maxBufferSize);
console.log('Max storage buffer:', device.limits.maxStorageBufferBindingSize);
```

Typical limits:
- Chrome: 2GB per buffer (4GB+ with memory64)
- Safari: 8GB+ per buffer on Apple Silicon
- Firefox: 256MB per buffer (limited)

## Benchmarking

### Running Performance Tests

```bash
# Unit tests (mock inference)
npm test -- --grep "Doppler Inference"

# E2E with real Doppler inference
DOPPLER=true npx playwright test tests/e2e/rsi-loop.spec.js --headed

# Specific model benchmark
TEST_MODEL=gemma3-4b-q4 DOPPLER=true npx playwright test tests/e2e/rsi-loop.spec.js
```

### Metrics Collected

The test suite measures:

1. **TTFT (Time-to-First-Token):** Prefill latency
2. **Decode Speed (tok/s):** Token generation rate
3. **Total Time:** End-to-end generation time
4. **Memory Usage:** Peak VRAM utilization

### Example Output

```
=== DOPPLER PERFORMANCE METRICS ===
{
  "model": "gemma3-1b-q4",
  "tier": "Unified Memory",
  "tierLevel": 1,
  "metrics": {
    "tokenCount": 100,
    "totalTimeMs": 1250,
    "ttftMs": 320,
    "tokPerSec": 80.0,
    "decodeTokPerSec": 106.4
  }
}
```

## Troubleshooting

### "Model too large for GPU"

- Check tier level and model requirements
- Use a smaller quantization (Q4 instead of F16)
- Try a smaller model

### "WebGPU not available"

- Update browser to latest version
- Check `chrome://gpu` or `about:gpu` for WebGPU status
- Try Chrome Canary for latest features

### "Out of memory during inference"

- Reduce `maxTokens` parameter
- Clear browser cache/OPFS storage
- Close other GPU-intensive tabs

### "Slow performance"

- Ensure hardware acceleration is enabled
- Check for thermal throttling
- Run `doppler test kernels --perf` to benchmark kernels

## See Also

- [Doppler Architecture](./ARCHITECTURE.md)
- [Kernel Compatibility](./KERNEL_COMPATIBILITY.md)
- [Testing Guide](./TESTING.md)
