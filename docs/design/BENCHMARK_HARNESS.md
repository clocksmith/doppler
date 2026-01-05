# DOPPLER Benchmark Harness Specification

Defines a standardized benchmark harness for DOPPLER so performance claims are measurable and comparable across devices, browsers, and competing runtimes.

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Kernel microbenchmarks | ✅ Implemented | `kernel-tests/tests/benchmarks/` |
| Pipeline benchmark harness | ✅ Implemented | `tests/benchmark/pipeline-benchmark.ts` |
| System benchmarks | ✅ Implemented | `tests/benchmark/system-benchmark.ts` |
| Standard prompts | ✅ Implemented | `tests/benchmark/prompts.ts` |
| JSON result schema | ✅ Implemented | `tests/benchmark/types.ts` |
| GPU timestamp queries | ✅ Implemented | Uses `gpu/profiler.ts` |
| GPU readback tracking | ✅ Implemented | Tracked in harness |
| Peak VRAM estimation | ✅ Implemented | Uses `gpu/buffer-pool.ts` |
| OPFS storage metrics | ✅ Implemented | Via Storage API |
| Results storage (IndexedDB) | ✅ Implemented | `tests/benchmark/results-storage.ts` |
| Results export (JSON) | ✅ Implemented | `tests/benchmark/results-storage.ts` |
| Results directory | ✅ Implemented | `tests/results/` |
| Comparison utilities | ✅ Implemented | `tests/benchmark/results-storage.ts` |
| CLI tool | ✅ Implemented | `cli/index.ts` |

### Claude Skill

Use `doppler-benchmark` skill (`.claude/skills/doppler-benchmark/SKILL.md`) for guided benchmarking.

---

## Goals

- Make performance claims reproducible across machines.
- Separate cold start vs warm start behavior.
- Report the bottlenecks that matter in browser inference: GPU submits, readback points, bandwidth, and memory use.
- Enable apples-to-apples comparisons against WebLLM and other browser runtimes using the same model and prompt set.

---

## Scope

The harness benchmarks three layers:

1. **Kernel microbench**: single-op timings (matmul, attention, dequant) with synthetic tensors.
   - Implemented in `kernel-tests/tests/benchmarks/`.
2. **Pipeline benchmarks**: prefill and decode loops using a real model manifest.
   - Implemented in `tests/benchmark/pipeline-benchmark.ts`.
3. **System benchmarks**: download and storage behavior (HTTP vs OPFS vs Native Bridge, and later P2P).
   - Implemented in `tests/benchmark/system-benchmark.ts`.

---

## Metrics (Required)

### Latency and Throughput

- `ttft_ms`: time from `generate()` start to first token emitted.
- `prefill_ms`: wall time for prefill forward pass (prompt processing).
- `decode_ms_total`: wall time for the decode loop (generated tokens only).
- `decode_ms_per_token_p50`, `decode_ms_per_token_p90`, `decode_ms_per_token_p99`: distribution over decode steps.
- `prefill_tokens_per_sec`: promptTokens / (prefill_ms / 1000).
- `decode_tokens_per_sec`: generatedTokens / (decode_ms_total / 1000).

### GPU Scheduling and Readback

- `gpu_submit_count_prefill`: number of `queue.submit()` calls during prefill.
- `gpu_submit_count_decode`: number of `queue.submit()` calls during decode.
- `gpu_readback_bytes_total`: total bytes copied GPU to CPU for the run (logits and any debug reads).
- `gpu_timestamp_available`: whether timestamp queries are supported.
- `gpu_time_ms_prefill`, `gpu_time_ms_decode`: GPU time if timestamp queries are enabled.

### Memory and Storage

- `estimated_vram_bytes_peak`: peak bytes allocated in buffer pool and persistent GPU buffers.
- `kv_cache_dtype`: `f16` or `f32`.
- `kv_cache_max_seq_len`: configured cache length.
- `storage_mode`: `opfs` or `native_bridge` or `http_only`.
- `storage_persisted`: result of `navigator.storage.persisted()`.
- `opfs_usage_bytes`: measured OPFS directory size when supported.

### Distribution (Cold Start)

- `origin_bytes_downloaded`: bytes fetched from HTTP origin (if applicable).
- `opfs_bytes_written`: bytes written to OPFS during model acquisition.
- `download_wall_ms`: wall time to populate local cache from origin.

P2P extension metrics are defined in Phase 4 roadmap.

---

## Benchmark Matrix (Required)

The harness must record environment metadata:

- Browser: name and version
- OS: name and version
- GPU: adapter info (vendor, device, description) and relevant WebGPU features
- WebGPU features: `shader-f16`, `subgroups`, `timestamp-query`
- Model: `modelId`, `quantization`, `totalSize`, `tensorCount`

Recommended minimum matrix:

- Browsers: Chrome, Safari (macOS), Firefox (if usable)
- GPUs: Apple Silicon (unified), AMD (Linux), NVIDIA (discrete)
- Models: one small dense (1B), one medium dense (3B-8B), one MoE (Mixtral or GPT-OSS)

---

## Workloads (Required)

### Standard Prompts

Use a small fixed set of prompts and record the tokenized lengths:

- `short`: 16-64 tokens
- `medium`: 256-512 tokens
- `long`: 2048 tokens (or nearest feasible length for model and browser limits)

Prompts should be deterministic text and stored in the repo (no network fetch during benchmark).

### Generation Settings

To maximize comparability:

- Default: `temperature = 0`, `topK = 1`, `topP = 1` (greedy) for deterministic decode.
- Report any deviation from greedy in the run metadata.

---

## Methodology (Required)

### Cold vs Warm Runs

Each benchmark suite runs:

- `cold`: OPFS empty (or model directory deleted), then download and load.
- `warm`: model already cached in OPFS, then load and run.

**CLI note:** When running via `cli/index.ts`, OPFS persistence depends on using a stable Playwright profile directory. Use `--profile-dir` to explicitly control this:

- `warm`: reuse the same `--profile-dir`
- `cold`: use a fresh `--profile-dir` (or delete the profile dir)

### Warmup

Perform warmup passes to avoid shader compilation skew:

- `warmup_prefill_runs`: 1-3
- `warmup_decode_tokens`: 8-16

### Measurement Rules

- Use `performance.now()` for wall clock.
- Avoid debug readbacks during timed sections unless explicitly measuring debug overhead.
- Report CPU-only fallbacks as invalid results for GPU benchmarks.

---

## Results Format (Required)

Write results as JSON so they can be compared automatically.

### Example Result JSON

```json
{
  "schemaVersion": 1,
  "timestamp": "2025-12-15T12:34:56Z",
  "suite": "pipeline",
  "runType": "warm",
  "env": {
    "browser": { "name": "Chrome", "version": "142.0.0.0" },
    "os": { "name": "Linux", "version": "6.17.0" },
    "gpu": { "vendor": "AMD", "device": "Radeon", "description": "Strix Halo" },
    "webgpu": { "hasF16": true, "hasSubgroups": true, "hasTimestampQuery": false }
  },
  "model": {
    "modelId": "dcc83e...",
    "quantization": "Q4_K_M",
    "totalSizeBytes": 965000000,
    "tensorCount": 340
  },
  "workload": {
    "promptName": "medium",
    "promptTokens": 384,
    "maxNewTokens": 128,
    "sampling": { "temperature": 0, "topK": 1, "topP": 1 }
  },
  "metrics": {
    "ttft_ms": 820,
    "prefill_ms": 760,
    "prefill_tokens_per_sec": 505,
    "decode_ms_total": 3120,
    "decode_tokens_per_sec": 41,
    "gpu_submit_count_prefill": 1,
    "gpu_submit_count_decode": 128,
    "gpu_readback_bytes_total": 512,
    "estimated_vram_bytes_peak": 3200000000
  }
}
```

---

## Competitor Comparison Policy

Comparisons must specify:

- Same model and quantization, or an explicit conversion mapping.
- Same prompt and tokenization behavior (report prompt token count).
- Same sampling settings.

For WebLLM comparisons, record:

- WebLLM version or commit
- Runtime configuration (model artifact, backend, and any flags)
- Any differences in caching or shader warmup behavior

---

## WebLLM Comparison Benchmark (Required for Claims)

To make credible performance claims against WebLLM, DOPPLER must benchmark against the **exact models and methodology** used in WebLLM's published results.

### Primary Benchmark Model

**Llama-3.1-8B-Instruct Q4** (q4f16_1)

This is the model from WebLLM's [arXiv paper (2412.15803)](https://arxiv.org/abs/2412.15803):

| Metric | WebLLM Result | Source |
|--------|---------------|--------|
| Decode speed | **41.1 tok/s** | arXiv paper, M3 Max |
| Native MLC-LLM | 57.7 tok/s | arXiv paper, M3 Max |
| % of native | 71.2% | Calculated |
| Prefill tokens | 64 | MLC blog methodology |
| Decode tokens | 128 | MLC blog methodology |
| Quantization | 4-bit weights, f16 compute | q4f16_1 |
| VRAM required | ~5GB | Estimated |

**WebLLM Model ID:** `Llama-3.1-8B-Instruct-q4f16_1-MLC`

**Live demo:** https://webllm.mlc.ai/

### Why Llama-3.1-8B?

| Factor | Llama-3.1-8B | Gemma 1B (current DOPPLER test) |
|--------|--------------|--------------------------------|
| Published WebLLM benchmark | **Yes (41.1 tok/s)** | No |
| Model size class | 8B (industry standard) | 1B (tiny) |
| Industry relevance | High (widely deployed) | Low (demo only) |
| Fair comparison | Direct, same model | Scaled estimate only |
| Apples-to-apples | **Yes** | No |

**Note:** Current DOPPLER benchmarks on Gemma 1B Q4 show 4-5 tok/s. A 1B model should be ~8x faster than an 8B model. If WebLLM achieves 41 tok/s on 8B, it would achieve ~300+ tok/s on 1B. This means DOPPLER is currently **~60x slower than expected** on equivalent workloads.

### Alternative Benchmark Models

| Model | WebLLM Benchmark | VRAM | Priority | Notes |
|-------|------------------|------|----------|-------|
| **Llama-3.1-8B Q4** | 41.1 tok/s | ~5GB | **P0** | Primary comparison target |
| Phi-3.5-mini 3.8B Q4 | 71.1 tok/s | ~3GB | P1 | Faster, good for lower VRAM |
| Llama-3.2-3B Q4 | ~90 tok/s (claimed) | ~2GB | P2 | Newer model, less benchmark data |
| Gemma-2-9B Q4 | Not published | ~6GB | P3 | Similar size, no WebLLM reference |

### Test Protocol (Must Match WebLLM Methodology)

To ensure fair comparison with WebLLM's published numbers:

```
Prefill:      64 tokens (fixed)
Decode:       128 tokens (fixed)
Warmup:       Discard first run
Measurement:  Second run (warm)
Sampling:     temperature=0, topK=1, topP=1 (greedy)
Quantization: Q4_K_M or q4f16 (equivalent 4-bit)
```

**Source:** [MLC Blog](https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine) - "64 prefill tokens, decoding 128 tokens"

### Hardware Requirements

For M3 MacBook comparison (matching WebLLM paper):

| Spec | WebLLM Paper | DOPPLER Test |
|------|--------------|--------------|
| Chip | M3 Max | M3 (any variant) |
| RAM | 36GB+ | 16GB+ (for 8B Q4) |
| Browser | Chrome (WebGPU) | Chrome 113+ |
| OS | macOS | macOS |

### Benchmark Checklist

Before claiming performance parity or superiority to WebLLM:

- [ ] **Model support:** DOPPLER can load and run Llama-3.1-8B Q4
- [ ] **Same quantization:** Q4_K_M matches q4f16 behavior
- [ ] **Same workload:** 64 prefill, 128 decode tokens
- [ ] **Same hardware:** M3 MacBook (specify variant)
- [ ] **Warm run:** Discard first run, measure second
- [ ] **Greedy sampling:** temperature=0, topK=1
- [ ] **Record submits:** Track `gpu_submit_count_decode` (target: 1 per token)
- [ ] **Compare decode tok/s:** DOPPLER vs WebLLM's 41.1 tok/s

### Current Blocker

**Does DOPPLER support Llama-3.1-8B?**

If not, priority actions:
1. Add Llama 3.1 8B architecture support to DOPPLER loader
2. Convert Llama-3.1-8B-Instruct to RDRR format
3. Run benchmark with WebLLM-matching test config
4. Compare directly to WebLLM's 41.1 tok/s baseline

If Llama support is complex, **Phi-3.5-mini 3.8B** (71.1 tok/s benchmark) is an acceptable interim target.

### Result Reporting

When reporting WebLLM comparison results:

```json
{
  "comparison": {
    "competitor": "WebLLM",
    "competitorVersion": "0.2.80",
    "competitorResult": {
      "model": "Llama-3.1-8B-Instruct-q4f16_1-MLC",
      "decode_tokens_per_sec": 41.1,
      "source": "arXiv:2412.15803",
      "hardware": "M3 Max MacBook Pro"
    },
    "dopplerResult": {
      "model": "llama-3.1-8b-instruct-q4",
      "decode_tokens_per_sec": null,
      "hardware": "M3 MacBook Pro"
    },
    "delta_percent": null,
    "methodology_match": true
  }
}
```

---

## Recommended Repo Layout (Non-binding)

- Kernel microbenchmarks: `kernel-tests/tests/benchmarks/`
- Pipeline benchmark harness: `tests/benchmark/`
- Saved result JSON: `tests/results/`

---

## Usage

### CLI (Recommended)

The CLI is the single entry point for running benchmarks (server auto-starts):

```bash
# Quick benchmark with xs prompt (headed browser)
doppler bench inference --prompt xs --headed

# Standard benchmarks
doppler bench inference                        # Headless (default: gemma3-1b-q4)
doppler bench inference --headed               # With visible browser window

# Custom options
doppler bench inference --prompt medium        # Different prompt size (xs/short/medium/long)
doppler bench inference --runs 3               # Multiple runs for statistics
doppler --help                                 # Show all CLI options
```

Results auto-save to `tests/results/{suite}_{model}_{timestamp}.json`

### Browser Console

For interactive benchmarking in the browser DevTools console:

### Quick Pipeline Benchmark

```typescript
import { runQuickBenchmark, formatBenchmarkSummary } from './tests/benchmark/index.js';

const result = await runQuickBenchmark('http://localhost:8080/models/gemma-3-1b-q4');
console.log(formatBenchmarkSummary(result));
console.log(JSON.stringify(result, null, 2));
```

### Full Pipeline Benchmark

```typescript
import { PipelineBenchmark } from './tests/benchmark/index.js';

const harness = new PipelineBenchmark({
  modelPath: 'http://localhost:8080/models/gemma-3-1b-q4',
  promptName: 'medium',
  maxNewTokens: 128,
  warmupRuns: 2,
  timedRuns: 3,
  sampling: { temperature: 0, topK: 1, topP: 1 },
});

const result = await harness.run();
```

### System Benchmark (Download/Storage)

```typescript
import { runSystemBenchmark, formatSystemSummary } from './tests/benchmark/index.js';

const result = await runSystemBenchmark('http://localhost:8080/models/gemma-3-1b-q4');
console.log(formatSystemSummary(result));
```

### Save and Compare Results

```typescript
import {
  saveResult,
  downloadAsJSON,
  loadResultsByModel,
  comparePipelineResults,
  formatComparison
} from './tests/benchmark/index.js';

// Save to IndexedDB
await saveResult(result);

// Download as JSON file
downloadAsJSON(result);

// Compare historical results
const history = await loadResultsByModel('gemma-3-1b-q4');
if (history.length >= 2) {
  const deltas = comparePipelineResults(history[0], history[1]);
  console.log(formatComparison(deltas));
}
```

### Available Prompts

| Name | Token Range | Use Case |
|------|-------------|----------|
| `xs` | 6-10 | Fast iteration ("The color of the sky is") |
| `short` | 16-64 | Quick validation |
| `medium` | 256-512 | Standard benchmark |
| `long` | ~2048 | Stress test |

---

*Last updated: December 20, 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--kernel-plan`, `--kernel-profile`), and the OPFS purge helper.

