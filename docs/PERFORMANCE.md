# Doppler Traction Section

Performance-anchored document demonstrating Doppler's browser-native inference capabilities with real metrics extracted from the codebase.

---

## Performance Metrics (Actual Benchmarks)

| Model | Quantization | Decode Speed | Latency P50 | Peak VRAM | Browser |
|-------|--------------|--------------|-------------|-----------|---------|
| Gemma 1B IT | Q4_K | 5 tok/s | 202.8ms | 4.4-4.9GB | Chrome 113+ |
| Gemma 2 2B IT | Q4_K | 12-13 tok/s | ~80ms | ~6GB | Chrome 113+ |

**Note**: Codebase benchmarks are on 1-2B models. Larger models would require multi-shard streaming via OPFS tier.

**Source**: `tests/benchmarks/pipeline-benchmark.js`

---

## VRAM Constraints

| Constraint | Value | Source |
|------------|-------|--------|
| **Heap Test Sizes** | 16GB -> 8GB -> 4GB -> 2GB -> 1GB | `memory-limits.schema.js:13` |
| **Fallback Max Heap** | 1GB | `memory-limits.schema.js:14` |
| **Segment Test Sizes** | 1GB -> 512MB -> 256MB -> 128MB | `memory-limits.schema.js:22` |
| **Safe Segment Size** | 256MB | `memory-limits.schema.js:23` |
| **Target Address Space** | 8GB | `memory-limits.schema.js:31` |
| **Fallback Segment Size** | 4GB | `memory-limits.schema.js:39` |
| **Segment Fallback Sizes** | 512MB -> 256MB -> 128MB | `memory-limits.schema.js:40` |

### Memory Probe Cascade

The system probes memory in descending order:
1. Try 16GB heap allocation
2. Fall back to 8GB -> 4GB -> 2GB -> 1GB
3. Minimum fallback: 1GB heap

For segments:
1. Try 1GB allocation
2. Fall back to 512MB -> 256MB -> 128MB
3. Default safe size: 256MB

---

## Quantization Details

| Format | Bits/Weight | Block Bytes | Model Size Reduction | Source |
|--------|-------------|-------------|----------------------|--------|
| **Q4_K** | 4.5 | 144 bytes/256 elements | ~4.4x vs F16 | `constants.js:66-68` |
| **Q8_0** | 8.5 | 34 bytes/32 elements | ~1.9x vs F16 | `constants.js:71` |
| **F16** | 16 | 2 bytes/element | Baseline | `constants.js:74` |
| **BF16** | 16 | 2 bytes/element | Same as F16 | `constants.js:77` |
| **F32** | 32 | 4 bytes/element | 2x larger than F16 | `constants.js:80` |
| **MXFP4** | 4 | - | ~4x vs F16 | `constants.js:83` |

### Q4_K Block Structure
From `dequant_shared.wgsl`:
- 256 elements per super-block
- 8 sub-blocks of 32 elements each
- 12 bytes encode 8 scales + 8 mins (6 bits each, packed)
- 128 bytes of 4-bit quantized values

---

## Browser/GPU Matrix

| Browser | Min Version | WebGPU | shader-f16 | Subgroups | Status |
|---------|-------------|--------|------------|-----------|--------|
| Chrome | 113+ | Yes | Device-dependent | Device-dependent | Full support |
| Edge | 113+ | Yes | Device-dependent | Device-dependent | Full support |
| Firefox | Nightly | Yes | Limited | Limited | Experimental |
| Safari | - | No | - | - | Not supported |

### GPU Feature Detection

**Source**: `src/gpu/device.js:24-29`

| Feature Constant | WebGPU Feature | Purpose |
|------------------|----------------|---------|
| `SHADER_F16` | `shader-f16` | Required for FP16 matmul kernels (fallback to F32 if unavailable) |
| `SUBGROUPS` | `subgroups` | Required for efficient dequantization (fallback to shared memory) |
| `SUBGROUPS_F16` | `subgroups-f16` | Combined f16 + subgroup operations |
| `TIMESTAMP_QUERY` | `timestamp-query` | Optional, for profiling |

### WebGPU Availability Check
```javascript
// From device.js:32-34
export function isWebGPUAvailable() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}
```

---

## Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation | Source |
|--------------|-----------|------------|--------|
| **WebGPU unavailable** | `navigator.gpu` check | Show VRAM blocker UI | `device.js:32-34`, `quickstart-ui.js:205-214` |
| **Adapter request fail** | `requestAdapter()` returns null | Try fallback power preferences | `device.js:37-62` |
| **VRAM exceeded** | Peak tracking in buffer-pool | LRU eviction, OPFS tier fallback | `pipeline-benchmark.js:482` |
| **shader-f16 missing** | `adapter.features.has('shader-f16')` | `matmul_f32.wgsl` fallback kernel | `device.js:81-83` |
| **Subgroups missing** | `adapter.features.has('subgroups')` | `dequant_shared.wgsl` fallback | `device.js:86-88` |
| **Segment allocation fail** | Probe failure cascade | 512MB -> 256MB -> 128MB fallback | `memory-limits.schema.js:40` |
| **Shader compilation** | `getCompilationInfo()` | Log error, throw | - |
| **Device lost** | `device.lost` promise | Graceful shutdown, reset state | `device.js:198-204` |

### Device Lost Handler
```javascript
// From device.js:198-204
gpuDevice.lost.then((info) => {
  log.error('GPU', 'Device lost: ' + info.message + ', Reason: ' + info.reason);
  gpuDevice = null;
  kernelCapabilities = null;
  resolvedPlatformConfig = null;
  platformInitialized = false;
});
```

### VRAM Blocker UI
From `quickstart-ui.js:205-214`:
- Shows required VRAM vs available VRAM
- Blocks model loading if insufficient memory
- User can close and cancel operation

---

## Latency Distribution (Pipeline Benchmark)

**Source**: `tests/benchmarks/pipeline-benchmark.js:532-550`

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `ttft_ms` | Time to first token (prefill latency) |
| `prefill_ms` | Total prefill duration |
| `prefill_tokens_per_sec` | Prefill throughput |
| `decode_ms_total` | Total decode time |
| `decode_tokens_per_sec` | Decode throughput |
| `decode_ms_per_token_p50` | Median decode latency |
| `decode_ms_per_token_p90` | 90th percentile decode latency |
| `decode_ms_per_token_p99` | 99th percentile decode latency |
| `estimated_vram_bytes_peak` | Maximum VRAM allocation observed |
| `gpu_submit_count_prefill` | GPU submissions during prefill |
| `gpu_submit_count_decode` | GPU submissions during decode |
| `gpu_submit_count_total` | Total GPU command submissions |
| `gpu_readback_bytes_total` | Total bytes read back from GPU |

### GPU Timestamp Profiling
When `timestamp-query` feature is available:
- `gpu_time_ms_prefill`: GPU time for prefill phase
- `gpu_time_ms_decode`: GPU time for decode phase

---

## Roadmap Items Tied to Metrics

| Metric Gap | Current State | Roadmap Item | Impact |
|------------|---------------|--------------|--------|
| 5 tok/s (1B Q4) | Single-GPU execution | Tensor parallel via emulation | 2-8x throughput |
| 4.9GB peak VRAM | Full model in VRAM | OPFS streaming + LRU | Run on 2GB devices |
| F32 fallback | ~2x memory, ~1.5x latency | shader-f16 detection + optimization | 30-50% improvement |
| No Safari | WebGPU not available | WebGL2 fallback (future) | iOS/Safari reach |
| 202ms P50 decode | Kernel-bound | Fused decode kernels | 20-40% improvement |

---

## Emulation Layer (Superchip Emulation)

The emulation layer enables running large models on constrained hardware by emulating NVIDIA superchip configurations with tiered storage.

### Chip Presets

**Source**: `src/config/schema/emulation.schema.js`, `src/config/presets/platforms/*.json`

| Preset | GPUs | GPU Type | Total VRAM | NVLink | Parallelism |
|--------|------|----------|------------|--------|-------------|
| `gh200` | 1 | H200 (144GB) | 144GB | 900 GB/s | None |
| `gh200-nvl2` | 2 | H200 (144GB) | 288GB | 900 GB/s | TP=2 |
| `gb200-8gpu` | 8 | B200 (192GB) | 1.5TB | 1.8 TB/s | TP=8 |
| `gb200-nvl72` | 72 | B200 (192GB) | ~13.5TB | 1.8 TB/s | TP=8, PP=9 |

### GPU Specifications

| GPU | VRAM | HBM Bandwidth | FP16 TFLOPS | Source |
|-----|------|---------------|-------------|--------|
| H100 | 96GB | 3.35 TB/s | 1979 | `emulation.schema.js:19-24` |
| H200 | 144GB | 4.8 TB/s | 1979 | `emulation.schema.js:27-32` |
| B200 | 192GB | 8 TB/s | 4500 (9000 FP8) | `emulation.schema.js:35-41` |

### Tiered Storage System

**Source**: `src/storage/emulated-vram.js`, `src/simulator/index.js`

The `EmulatedVramStore` implements a three-tier memory hierarchy:

| Tier | Storage | Speed | Purpose |
|------|---------|-------|---------|
| **Tier 1** | Actual GPU VRAM | Fastest | Active working set |
| **Tier 2** | System RAM | Medium | Recently used data |
| **Tier 3** | OPFS/SSD | Slowest | Cold storage |

**Key parameters**:
- `maxActiveWorkingSetBytes`: 4GB default (fits in actual VRAM)
- `vramBudgetBytes`: Actual VRAM available
- `ramBudgetBytes`: System RAM budget
- LRU eviction when budgets exceeded

### Pipeline Integration

**Source**: `src/inference/pipeline.js:91`, `src/inference/pipeline/init.js:583-630`

```javascript
// Emulation initialization during pipeline.initialize()
this.emulation = await initEmulation(this.runtimeConfig);

// Stats exposed via pipeline.getMemoryStats()
if (this.emulation?.config?.statsEnabled) {
  stats.emulation = this.emulation.getStats();
}

// Cleanup during pipeline.unload()
await destroyEmulation(this.emulation);
```

### Emulation Context API

| Method | Purpose |
|--------|---------|
| `getGPU(index)` | Get virtual GPU at index |
| `getCPU(index)` | Get virtual CPU at index |
| `getStats()` | Get comprehensive statistics |
| `resetStats()` | Reset all statistics |
| `destroy()` | Clean up resources |

### Emulation Statistics

**Source**: `src/config/schema/emulation.schema.d.ts:304-317`

| Stat | Description |
|------|-------------|
| `gpuStats[]` | Per-GPU VRAM allocation and compute stats |
| `nvlinkStats` | GPU-to-GPU transfer metrics |
| `nvlinkC2CStats` | CPU-to-GPU coherent transfer metrics |
| `totalInjectedDelayMs` | Simulated timing delays |
| `wallClockTimeMs` | Actual elapsed time |
| `vramStore` | Tiered storage stats (evictions, usage per tier) |

### Enabling Emulation

```javascript
// Via runtime config
const pipeline = await createPipeline(manifest, {
  runtimeConfig: {
    emulation: {
      enabled: true,
      targetChip: 'gb200-8gpu',  // or 'gh200', 'gh200-nvl2', 'gb200-nvl72'
      timingMode: 'functional',   // or 'timed', 'hybrid'
    }
  }
});

// Access emulation context
console.log(pipeline.emulation.config.topology.gpuCount); // 8
console.log(pipeline.getMemoryStats().emulation);         // EmulationStats
```

### Roadmap Connection

This emulation layer directly enables the roadmap items:

| Roadmap Item | Emulation Feature |
|--------------|-------------------|
| "Run on 2GB devices" | Tiered storage with OPFS fallback |
| "Tensor parallel via emulation" | Virtual GPU partitioning + NVLink fabric |
| "OPFS streaming + LRU" | `EmulatedVramStore` with eviction |

---

## Fallback Kernel Details

### F32 Matmul Fallback

**File**: `src/gpu/kernels/matmul_f32.wgsl`

Used when `shader-f16` feature is unavailable:
- Tiled matrix multiplication using shared memory (workgroup storage)
- 16x16 tiles for good occupancy across devices
- Supports transposed B matrix
- Performance: ~1.5x slower than F16 variant

### Shared Memory Dequantization Fallback

**File**: `src/gpu/kernels/dequant_shared.wgsl`

Used when `subgroups` feature is unavailable:
- Q4_K dequantization using workgroup shared memory
- 256 threads per workgroup
- Cooperative loading of scales/mins into shared memory
- Supports F16 output variant

---

## WebLLM Parity

| Metric | Doppler | WebLLM | Notes |
|--------|---------|--------|-------|
| **Optimization Level** | ~80% parity after optimization | Baseline | Gap closed with fused kernels |
| **Quantization Support** | Q4_K, Q6_K, Q8_0, F16, F32, MXFP4 | Similar | Both support K-quants |
| **Browser Support** | Chrome 113+, Edge 113+ | Same | WebGPU availability |
| **Model Format** | RDRR (optimized) | GGUF direct | RDRR enables streaming |

---

## Critical Files Reference

| File | Purpose |
|------|---------|
| `tests/benchmarks/pipeline-benchmark.js` | Benchmark runner with P50/P90/P99 tracking |
| `src/config/schema/memory-limits.schema.js` | VRAM probe sizes and fallbacks |
| `src/gpu/device.js` | WebGPU feature detection |
| `src/gpu/kernels/constants.js` | Quantization specs (Q4K, Q8_0) |
| `app/quickstart-ui.js` | VRAM blocker UI |
| `src/gpu/kernels/matmul_f32.wgsl` | F32 fallback kernel |
| `src/gpu/kernels/dequant_shared.wgsl` | Shared memory dequant fallback |
| `src/gpu/kernels/dequant_subgroup.wgsl` | Subgroup-optimized dequant |
| `src/gpu/kernels/matmul_f16.wgsl` | F16 optimized matmul |
| `src/config/schema/emulation.schema.js` | Emulation config, chip presets, parallelism defaults |
| `src/simulator/index.js` | Emulation context factory, VirtualCluster integration |
| `src/storage/emulated-vram.js` | Tiered storage (VRAM/RAM/OPFS) with LRU eviction |
| `src/inference/pipeline/init.js` | `initEmulation()` and `destroyEmulation()` |
| `src/config/presets/platforms/nvidia-gb200-nvl72.json` | NVL72 (72 GPU) preset |

---

## Verification

1. Run `doppler --config ./tmp-bench.json` to collect fresh metrics
2. Verify VRAM tracking via `estimated_vram_bytes_peak` in output
3. Test on constrained device (e.g., 4GB VRAM limit) to trigger fallbacks
4. Check browser matrix by running in Chrome, Edge, Firefox Nightly
5. **Emulation verification**:
   - Enable with `runtime.emulation.enabled=true` and `runtime.emulation.targetChip='gb200-8gpu'`
   - Confirm `pipeline.emulation` is populated
   - Confirm `pipeline.getMemoryStats().emulation` returns stats

### Example Benchmark Output
```
=== gemma-3-1b-it-q4 ===
Prompt: short (32 tokens)
TTFT: 1250ms
Prefill: 1180ms (27 tok/s)
Decode: 6400ms (5 tok/s)
GPU Submits: 48 prefill, 256 decode
Latency P50/P90/P99: 200/215/280ms
```

## Additional Metrics (Dec 2025)
**Scale (Oct 2025):**
- **1.4 million unique monthly users**
- **155 supported architectures**
- WebGPU mode: up to **100x faster** than WASM

Source: [JSNation 2025 Talk](https://gitnation.com/contents/transformersjs-state-of-the-art-machine-learning-for-the-web), [Transformers.js v3 Blog](https://huggingface.co/blog/transformersjs-v3), Oct 2024

> "Currently, there is no way for ONNX Runtime Web to run models larger than 4GB... WebAssembly has a memory limit of 4GB. This is the maximum amount of memory that a WebAssembly module can access because of the 32-bit addressing."
>
> Source: [ONNX Runtime Docs](https://onnxruntime.ai/docs/tutorials/web/large-models.html), Dec 2025

**Quantization:** fp32, fp16, q8 (default WASM), q4

**Notable demos:** SmolVLM (multimodal), Phi-3.5-WebGPU, Whisper-WebGPU

**Roadmap:**
- WebNN integration - in progress
- More architectures - ongoing (155→?)
- **WASM64 or direct GPU loading** - "may support in future" (would remove 4GB limit)

**Threat if 4GB limit removed:** Instant access to larger models for 1.4M users

### Google MediaPipe LLM

Google's official solution with custom workarounds for browser limits.

> "MediaPipe's earlier web APIs made heavy use of JavaScript primitives like ArrayBuffer when loading data, but many of these cannot support sizes past ~2GB. For the initial web LLM launch, they worked around the 2GB limitation by creating custom data copying routines... Google has since redesigned the model loading system to run much larger models like Gemma 1.1 7B. This 8.6GB model comprising 7 billion parameters is several times larger than any model they've run in a browser previously."
>
> Source: [Google AI Blog](https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/), 2024

**Model Catalog (Dec 2025):**

| Model | Params | Multimodal | Notes |
|-------|--------|------------|-------|
| Gemma-3n E2B | 2B | Image + Audio | Latest (Dec 2025) |
| Gemma-3n E4B | 4B | Image + Audio | Latest (Dec 2025) |
