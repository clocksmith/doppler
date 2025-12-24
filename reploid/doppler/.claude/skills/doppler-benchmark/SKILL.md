---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks. Use when measuring inference speed, comparing against baselines, or tracking performance regressions. Outputs JSON results per the BENCHMARK_HARNESS spec.
---

# DOPPLER Benchmark Skill

You are running performance benchmarks for DOPPLER, a browser-native WebGPU LLM inference engine.

## Resources

1. **Benchmark Spec**: `doppler/reploid/doppler/docs/spec/BENCHMARK_HARNESS.md`
   - Required metrics (TTFT, prefill, decode throughput)
   - JSON output schema
   - Methodology (cold vs warm, warmup, measurement rules)

2. **Benchmark Harness**: `doppler/reploid/doppler/tests/benchmark/`
   - `pipeline-benchmark.ts` - Main harness class
   - `types.ts` - TypeScript interfaces for results
   - `prompts.ts` - Standard test prompts (short/medium/long)

3. **Kernel Benchmarks**: `doppler/reploid/doppler/kernel-tests/tests/benchmarks/`
   - `matmul.bench.ts` - Matrix multiplication
   - `all-kernels.bench.ts` - Full kernel suite
   - `config.ts` - Workload configurations

## Quick Start (CLI)

The CLI handles building, server startup, and benchmarks automatically. From `doppler/reploid/`:

```bash
# Quick benchmark with xs prompt (headed) - fastest iteration
doppler bench inference --prompt xs --headed

# Standard benchmarks
doppler bench inference                        # Headless (default: gemma3-1b-q4)
doppler bench inference --headed               # With visible browser window

# Custom options
doppler bench inference --headed --prompt medium    # Different prompt size
doppler bench inference --prompt short --runs 3     # Multiple runs for statistics
doppler --help                                      # Show all options
```

**CLI Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m` | Model name in models/ | gemma3-1b-q4 |
| `--suite, -s` | quick, pipeline, full, system | pipeline |
| `--prompt, -p` | xs, short, medium, long | medium |
| `--runs, -r` | Number of timed runs | 3 |
| `--warmup, -w` | Number of warmup runs | 2 |
| `--max-tokens, -t` | Max tokens to generate | 128 |
| `--output, -o` | Additional JSON output path | (none) |
| `--html` | Custom HTML report path | auto-generated |
| `--compare, -c` | Baseline JSON for comparison | (none) |
| `--retries` | Retry failed runs | 2 |
| `--verbose, -v` | Show all browser logs | false |
| `--quiet, -q` | Suppress JSON to stdout | false |
| `--headed` | Show browser window | false |
| `--debug` | Enable debug GPU readbacks | false |

**Debug Flag Note:** The `--debug` flag enables verbose layer-by-layer GPU readbacks. This significantly slows benchmarks (adds GPU sync points). Only use for debugging, not performance measurement.

**Auto-generated outputs:**
```
tests/results/
├── pipeline_gemma-1b_apple-m1_2024-01-15T10-30-00.json   # Raw data
└── pipeline_gemma-1b_apple-m1_2024-01-15T10-30-00.html   # Visual report
```

## Quick Start (Browser Console)

```typescript
// Quick pipeline benchmark
import { runQuickBenchmark, formatBenchmarkSummary } from './tests/benchmark/index.js';
const result = await runQuickBenchmark('http://localhost:8080/models/gemma-1b');
console.log(formatBenchmarkSummary(result));

// Full pipeline benchmark with config
import { PipelineBenchmark } from './tests/benchmark/index.js';
const harness = new PipelineBenchmark({
  modelPath: 'http://localhost:8080/models/gemma-1b',
  promptName: 'medium',
  maxNewTokens: 128,
  warmupRuns: 2,
  timedRuns: 3,
  sampling: { temperature: 0, topK: 1, topP: 1 },
});
const result = await harness.run();
console.log(JSON.stringify(result, null, 2));

// System benchmark (download/storage)
import { runSystemBenchmark, formatSystemSummary } from './tests/benchmark/index.js';
const sysResult = await runSystemBenchmark('http://localhost:8080/models/gemma-1b');
console.log(formatSystemSummary(sysResult));

// Save results to IndexedDB
import { saveResult, downloadAsJSON } from './tests/benchmark/index.js';
await saveResult(result);
downloadAsJSON(result); // Downloads JSON file

// Compare results
import { loadResultsByModel, comparePipelineResults, formatComparison } from './tests/benchmark/index.js';
const history = await loadResultsByModel('gemma-1b');
if (history.length >= 2) {
  const deltas = comparePipelineResults(history[0], history[1]);
  console.log(formatComparison(deltas));
}
```

## Metrics Collected

### Pipeline Metrics

| Metric | Description |
|--------|-------------|
| `ttft_ms` | Time to first token |
| `prefill_ms` | Prefill wall time |
| `prefill_tokens_per_sec` | Prefill throughput |
| `decode_tokens_per_sec` | Decode throughput |
| `decode_ms_per_token_p50/p90/p99` | Latency percentiles |
| `gpu_submit_count_*` | GPU command submissions |
| `gpu_readback_bytes_total` | Bytes read from GPU |
| `gpu_time_ms_*` | GPU time (if timestamp query available) |
| `estimated_vram_bytes_peak` | Peak VRAM usage |

### System Metrics

| Metric | Description |
|--------|-------------|
| `storage.mode` | Storage mode (opfs/http_only) |
| `storage.quotaBytes` | Total storage quota |
| `download.bytesPerSec` | HTTP download speed |
| `opfs.writeBytesPerSec` | OPFS write speed |
| `opfs.readBytesPerSec` | OPFS read speed |

## Standard Prompts

| Name | Token Range | Use Case |
|------|-------------|----------|
| `xs` | 6-10 | Fast iteration ("The color of the sky is") |
| `short` | 16-64 | Quick validation |
| `medium` | 256-512 | Standard benchmark |
| `long` | ~2048 | Stress test |

## Workflow

**CLI (Recommended):**
1. Run: `doppler bench inference --prompt xs --headed` (quick) or `doppler bench inference` (full)
2. Results auto-save to `tests/results/` (both JSON + HTML report)

   Server auto-starts if not running.

**Browser Console (Manual):**
1. Start the dev server: `npm start`
2. Open browser to `http://localhost:8080`
3. Open DevTools console
4. Import and run benchmark harness
5. Save JSON results for comparison

## Comparing Results & Statistical Significance

**A/B Testing with Welch's t-test:**
```bash
# Save baseline
doppler bench inference -o baseline.json

# Make changes, then compare
doppler bench inference --compare baseline.json

# Output includes:
# - Delta% for each metric (TTFT, throughput, latency)
# - Welch's t-test on decode latencies
# - Statistical significance (p < 0.05)
```

**Example output:**
```
COMPARISON VS BASELINE
============================================================
TTFT                       45.0 ->       42.0  -6.7% ↓ BETTER
Decode tok/s               38.0 ->       41.0  +7.9% ↑ BETTER

STATISTICAL SIGNIFICANCE (Welch's t-test)
────────────────────────────────────────────────────────────
Decode Latency: t=-2.45, df=18.3, p=0.0243 (SIGNIFICANT)
  -> The difference IS statistically significant (p < 0.05)
```

**HTML Report Features:**
- Summary cards with key metrics
- SVG line chart of decode latency per token
- Baseline vs current bar charts (when comparing)
- Latency percentiles table
- Environment info (browser, GPU, OS)

**Comparing with WebLLM or other runtimes:**
- Use same model and quantization
- Use same prompt (record token count)
- Use greedy sampling (temperature=0)
- Record competitor version and config

## Quick Debug Benchmarks

Run benchmarks and filter output:

```bash
# Quick benchmark with xs prompt (uses "The color of the sky is")
doppler bench inference --prompt xs --headed

# With verbose output filtering
doppler bench inference --prompt xs --verbose 2>&1 | grep -E "logits|top-5|sampled" | head -20

# Check layer outputs
doppler bench inference --prompt xs --verbose 2>&1 | grep -E "LAYER_0|LAYER_25|FFN_OUTPUT" | head -30
```

## Related Skills

- **doppler-debug**: For investigating inference issues found during benchmarking
- **model-convert**: For preparing models in RDRR format for benchmarking

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

