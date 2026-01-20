# DOPPLER Benchmark Style Guide

Benchmarking conventions for DOPPLER. Benchmarks are test harnesses, not runtime code.

---

## Output Schema

- Emit JSON that conforms to `../spec/BENCHMARK_SCHEMA.json`.
- Always include `schemaVersion`, `timestamp`, and `suite`.
- Include `env`, `model`, `config`, `workload`, `metrics`, `quality`, and `raw` when available.

---

## Baseline Comparison

- Use `cli.compare` in config for regression checks.
- Respect `runtime.shared.benchmark.comparison.regressionThresholdPercent`.
- Fail the run when `failOnRegression` is enabled and any metric regresses beyond the threshold.

Baseline registry rules live under `runtime.shared.benchmark.baselines`.

---

## Run Policy

- Keep warmup and timed run counts in `runtime.shared.benchmark.run`.
- Use `runtime.shared.benchmark.stats` for outlier filtering, warmup stability, and thermal detection thresholds.
- Set `runtime.shared.tooling.intent = "calibrate"` for baseline benchmarks.
  If profiling, tracing, or probes are required, switch intent to `investigate`.

---

## Stats

- Use `src/debug/stats.js` for percentiles, IQR outlier removal, and confidence intervals.
- Avoid duplicate stats implementations in test harnesses.

---

## Profiling

Profiling is an investigation workflow. Do not profile while calibrating.

- Use `gpu/profiler.js` for GPU timestamps (not ad-hoc timers).
- Keep CPU timing in benchmark harnesses as a fallback.

---

## Benchmark Harness Specification (Merged)

Defines a standardized benchmark harness for DOPPLER so performance claims are measurable and comparable across devices, browsers, and competing runtimes.

### Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Kernel microbenchmarks | ✓ Implemented | `tests/kernels/benchmarks/` |
| Pipeline benchmark harness | ✓ Implemented | `tests/benchmarks/pipeline-benchmark.js` |
| System benchmarks | ✓ Implemented | `tests/benchmarks/system-benchmark.js` |
| Standard prompts | ✓ Implemented | `tests/benchmarks/prompts.js` |
| JSON result schema | ✓ Implemented | `../spec/BENCHMARK_SCHEMA.json` |
| GPU timestamp queries | ✓ Implemented | Uses `gpu/profiler.js` |
| GPU readback tracking | ✓ Implemented | Tracked in harness |
| Peak VRAM estimation | ✓ Implemented | Uses `memory/buffer-pool.js` |
| Output quality check | ✓ Implemented | `tests/benchmarks/pipeline-benchmark.js` |
| Baseline registry checks | ✓ Implemented | `tests/baselines.json` + CLI |
| OPFS storage metrics | ✓ Implemented | Via Storage API |
| Results storage (IndexedDB) | ✓ Implemented | `tests/benchmarks/results-storage.js` |
| Results export (JSON) | ✓ Implemented | `tests/benchmarks/results-storage.js` |
| Results directory | ✓ Implemented | `tests/results/` |
| Comparison utilities | ✓ Implemented | `tests/benchmarks/results-storage.js` |
| CLI tool | ✓ Implemented | `cli/index.js` |

### Claude Skill

Use `doppler-bench` skill (`../../.claude/skills/doppler-bench/SKILL.md`) for guided benchmarking.

---

## Config

Benchmark defaults live in `runtime.shared.benchmark` (see `src/config/schema/benchmark.schema.js`).
Use `extends: "bench"` in config; runtime config is the source of truth.
Baseline registry settings live under `runtime.shared.benchmark.baselines`.

---

## Goals

- Make performance claims reproducible across machines.
- Separate cold start vs warm start behavior.
- Report the bottlenecks that matter in browser inference: GPU submits, readback points, bandwidth, and memory use.
- Enable apples-to-apples comparisons against other browser runtimes using the same model and prompt set.

---

## Scope

The harness benchmarks three layers:

1. **Kernel microbench**: single-op timings (matmul, attention, dequant) with synthetic tensors.
2. **Pipeline benchmarks**: prefill and decode loops using a real model manifest.
3. **System benchmarks**: download and storage behavior (HTTP vs OPFS vs Native Bridge, and later P2P).

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
- `estimated_vram_bytes_peak_requested`: peak requested bytes before bucketing (sanity check).
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

When running via CLI, OPFS persistence depends on using a stable Playwright profile directory. Use `cli.profileDir` to explicitly control this:

- `warm`: reuse the same `cli.profileDir`
- `cold`: use a fresh `cli.profileDir` (or delete the profile dir)

### Warmup

Perform warmup passes to avoid shader compilation skew.

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
  "config": {
    "chain": ["bench", "default"],
    "runtime": { "...": "runtime config snapshot" },
    "benchmark": {
      "promptName": "medium",
      "customPrompt": null,
      "maxNewTokens": 128,
      "warmupRuns": 2,
      "timedRuns": 3,
      "sampling": { "temperature": 0, "topK": 1, "topP": 1 },
      "debug": false,
      "profile": false,
      "useChatTemplate": null
    }
  },
  "workload": {
    "promptName": "medium",
    "promptTokens": 384,
    "maxNewTokens": 128,
    "sampling": { "temperature": 0, "topK": 1, "topP": 1 }
  },
  "quality": {
    "ok": true,
    "reasons": [],
    "warnings": [],
    "stats": {
      "totalRuns": 3,
      "totalTokens": 128,
      "uniqueTokens": 110,
      "uniqueRatio": 0.859,
      "mostFrequentRatio": 0.12,
      "replacementChars": 0,
      "controlChars": 0
    }
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
    "estimated_vram_bytes_peak": 3200000000,
    "estimated_vram_bytes_peak_requested": 3100000000
  }
}
```

---

## Competitor Comparison Policy

Comparisons must specify:

- Same model and quantization, or an explicit conversion mapping.
- Same prompt and tokenization behavior (report prompt token count).
- Same sampling settings.

---

## WebLLM Comparison Benchmark (Required for Claims)

To make credible performance claims against WebLLM, DOPPLER must benchmark against the exact models and methodology used in WebLLM's published results.

### Primary Benchmark Model

**Llama-3.1-8B-Instruct Q4** (q4f16_1)

This is the model from WebLLM's arXiv paper (2412.15803):

| Metric | WebLLM Result | Source |
|--------|---------------|--------|
| Decode speed | 41.1 tok/s | arXiv paper, M3 Max |
| Native MLC-LLM | 57.7 tok/s | arXiv paper, M3 Max |
| % of native | 71.2% | Calculated |
| Prefill tokens | 64 | MLC blog methodology |
| Decode tokens | 128 | MLC blog methodology |
| Quantization | 4-bit weights, f16 compute | q4f16_1 |
| VRAM required | ~5GB | Estimated |

WebLLM model ID: `Llama-3.1-8B-Instruct-q4f16_1-MLC`

Live demo: https://webllm.mlc.ai/

### Why Llama-3.1-8B?

| Factor | Llama-3.1-8B | Gemma 1B (current DOPPLER test) |
|--------|--------------|----------------------------------|
| Published WebLLM benchmark | Yes (41.1 tok/s) | No |
| Model size class | 8B (industry standard) | 1B (tiny) |
| Industry relevance | High (widely deployed) | Low (demo only) |
| Fair comparison | Direct, same model | Scaled estimate only |
| Apples-to-apples | Yes | No |

Note: Current DOPPLER benchmarks on Gemma 1B Q4 show 4-5 tok/s. A 1B model should be ~8x faster than an 8B model. If WebLLM achieves 41 tok/s on 8B, it would achieve ~300+ tok/s on 1B. This means DOPPLER is currently ~60x slower than expected on equivalent workloads.

### Alternative Benchmark Models

| Model | WebLLM Benchmark | VRAM | Priority | Notes |
|-------|------------------|------|----------|-------|
| Llama-3.1-8B Q4 | 41.1 tok/s | ~5GB | P0 | Primary comparison target |
| Phi-3.5-mini 3.8B Q4 | 71.1 tok/s | ~3GB | P1 | Faster, good for lower VRAM |
| Llama-3.2-3B Q4 | ~90 tok/s (claimed) | ~2GB | P2 | Newer model, less benchmark data |
| Gemma-2-9B Q4 | Not published | ~6GB | P3 | Similar size, no WebLLM reference |

### Test Protocol (Must Match WebLLM Methodology)

```
Prefill:      64 tokens (fixed)
Decode:       128 tokens (fixed)
Warmup:       Discard first run
Measurement:  Second run (warm)
Sampling:     temperature=0, topK=1, topP=1 (greedy)
Quantization: Q4_K_M or q4f16 (equivalent 4-bit)
```

Source: https://blog.mlc.ai/2024/06/13/webllm-a-high-performance-in-browser-llm-inference-engine

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

- Model support: DOPPLER can load and run Llama-3.1-8B Q4
- Same quantization: Q4_K_M matches q4f16 behavior
- Same workload: 64 prefill, 128 decode tokens
- Same hardware: M3 MacBook (specify variant)
- Warm run: discard first run, measure second
- Greedy sampling: temperature=0, topK=1
- Record submits: target 1 decode submit per token
- Compare decode tok/s: DOPPLER vs WebLLM's 41.1 tok/s

### Result Reporting

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

- Kernel microbenchmarks: `tests/kernels/tests/benchmarks/`
- Pipeline benchmark harness: `tests/benchmark/`
- Saved result JSON: `tests/results/`

---

## Usage

### CLI (Recommended)

The CLI is the single entry point for running benchmarks (server auto-starts).
Command, suite, model id, and harness options live in config:

```bash
doppler --config ./tmp-bench.json
doppler --config ./tmp-bench-xs.json
doppler --help
```

Results auto-save to `tests/results/{suite}_{model}_{timestamp}.json`.

### Browser Console

Quick pipeline benchmark:

```typescript
import { runQuickBenchmark, formatBenchmarkSummary } from './tests/benchmark/index.js';

const result = await runQuickBenchmark('http://localhost:8080/models/gemma-3-1b-q4');
console.log(formatBenchmarkSummary(result));
console.log(JSON.stringify(result, null, 2));
```

Full pipeline benchmark:

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

System benchmark (download/storage):

```typescript
import { runSystemBenchmark, formatSystemSummary } from './tests/benchmark/index.js';

const result = await runSystemBenchmark('http://localhost:8080/models/gemma-3-1b-q4');
console.log(formatSystemSummary(result));
```

Save and compare results:

```typescript
import {
  saveResult,
  downloadAsJSON,
  loadResultsByModel,
  comparePipelineResults,
  formatComparison
} from './tests/benchmark/index.js';

await saveResult(result);
downloadAsJSON(result);

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
