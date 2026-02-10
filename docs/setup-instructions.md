# DOPPLER Setup (Browser WebGPU path)

DOPPLER is the local WebGPU inference engine for medium/large models. It uses the `.rpl` format (manifest + shard_*.bin) and supports three ways to load models.

For Ollama and other local server options, see [local-models.md](../../docs/local-models.md).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | WebGPU-capable | Discrete GPU with 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Storage | 4GB free | 20GB+ for multiple models |
| Browser | Chrome 113+ | Chrome/Edge with WebGPU enabled |

**Notes:**
- Unified memory (Apple Silicon, AMD Strix) is ideal for dense models
- Discrete GPUs benefit from MoE architectures or smaller shards
- Check `chrome://gpu` for WebGPU status

## WebGPU Tier System

Doppler detects GPU capabilities and assigns a tier level to guide model sizing.

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
# Start a static server
python3 -m http.server 8080

# Run benchmarks in the demo diagnostics UI (/demo/)
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

---

## Setup Methods

1) Serve local models (all browsers)
- Place an RDRR model under `/models/<model-id>/` with `manifest.json` and shard files.
- Serve the repo root with `python3 -m http.server 8080`.
- In the diagnostics UI (`/demo/`), pick the model from the local list.

2) Import GGUF or safetensors in-browser (Chrome/Edge)
- Open the demo UI (`/demo/`) and use the import flow.
- Streams GGUF/safetensors → RDRR directly into OPFS with progress UI. No CLI/server needed.

3) Convert via remote URLs
- Use the demo UI URL import to stream remote safetensors shards (with HTTP range + download fallback).
- Converted output is cached in OPFS.

Notes:
- Manifests include tensor locations and shard hashes; `hashAlgorithm` may be `sha256` or `blake3`.
- Unified memory (Apple/Strix) is ideal for dense models; discrete GPUs benefit from MoE or smaller shards.

## Troubleshooting

### WebGPU Not Available

1. Check browser version (Chrome 113+ required)
2. Enable WebGPU flag: `chrome://flags/#enable-unsafe-webgpu`
3. Check GPU status: `chrome://gpu`
4. Update GPU drivers
5. Try Firefox Nightly with WebGPU flag

### Model Loading Failed

| Symptom | Solution |
|---------|----------|
| "Out of memory" | Try smaller model or close other tabs |
| "Shader compilation failed" | Update GPU drivers, check WebGPU status |
| "Network error" | Check CORS settings on the static server or remote host |
| "Hash mismatch" | Re-download model, check disk integrity |

### Model Too Large for GPU

- Check tier level and model requirements
- Use a smaller quantization (Q4 instead of F16)
- Try a smaller model

### Out of Memory During Inference

- Reduce the `maxTokens` parameter
- Clear browser cache/OPFS storage
- Close other GPU-intensive tabs

### Slow Performance

- Ensure hardware acceleration is enabled
- Check for thermal throttling
- Open `/tests/kernels/browser/test-page.js` in a WebGPU browser to benchmark kernels

### Storage Management

- Models live in OPFS/IndexedDB
- Clear via: DevTools → Application → Storage → Clear site data
- Check usage: `navigator.storage.estimate()`

---

## Related Documentation

- [local-models.md](../../docs/local-models.md) - Ollama and server-based local models
- [quick-start.md](../../docs/quick-start.md) - General setup notes and common issues
- [security.md](../../docs/security.md) - Security considerations for local execution

---

*Last updated: January 2026*
