---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks to measure throughput, compare against baselines, and detect regressions across models or kernel settings. Use when validating speed changes or collecting JSON benchmark artifacts. (project)
---

# DOPPLER Benchmark Skill

This skill guides systematic performance measurement of DOPPLER inference.

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `decode_tokens_per_sec` | Main throughput metric | Higher is better |
| `ttft_ms` | Time to first token | Lower is better, measures prefill latency |
| `prefill_tokens_per_sec` | Prompt processing speed | Higher is better for long prompts |
| `decode_ms_per_token_p99` | 99th percentile per-token decode latency (ms) | Flag if >100ms for interactive use |
| `estimated_vram_bytes_peak` | Peak GPU memory usage (bytes) | Lower means more headroom |

## Standard Benchmark Commands

```bash
# Quick benchmark - single run
npm run bench -- -m MODEL 2>&1 | grep -E "TTFT|Prefill|Decode|GPU Submits"

# Multiple runs for statistical confidence (recommended)
npm run bench -- -m MODEL --runs 3 2>&1 | sed '/Benchmark complete/q'

# Save results to JSON for later comparison
npm run bench -- -m MODEL --runs 3 -o results.json

# Compare against saved baseline
npm run bench -- -m MODEL --compare baseline.json
```

## Fast Iteration Pattern

```bash
# Quick single run (minimal warmup)
npm run bench -- -m MODEL --runs 1 --warmup 0 2>&1 | sed '/Benchmark complete/q'

# After code changes: rebuild then benchmark
npm run build && npm run bench -- -m MODEL --runs 1 --warmup 0 2>&1 | sed '/Benchmark complete/q'
```

The `sed '/Benchmark complete/q'` pattern exits immediately after benchmark completes.

## Regression Detection Protocol

When asked "Is this slower?" or investigating performance regressions:

1. **Establish baseline** (if not already saved; use a clean worktree):
   ```bash
   git checkout main
   npm run bench -- -m MODEL --runs 3 -o baseline.json
   ```

2. **Benchmark current code**:
   ```bash
   npm run bench -- -m MODEL --runs 3 --compare baseline.json
   ```

3. **Interpret results**:
   - **<5% difference**: Within noise, not significant
   - **5-10% difference**: Investigate, may be real
   - **>10% difference**: Significant regression, needs fix

## Benchmark Configuration

For consistent results, use the `bench` config preset:

```bash
# Uses silent output, deterministic sampling, longer timeout
npm run bench -- --config bench -m MODEL
```

Or create a custom config for CI:

```json
{
  "extends": "bench",
  "runtime": {
    "debug": {
      "logOutput": { "stdout": false, "file": "./ci-logs/bench.log" }
    }
  },
  "cli": { "timeout": 300000 }
}
```

## Interpretation Guidelines

| Scenario | What to Check |
|----------|---------------|
| Prefill slow | Memory bandwidth, embedding gather |
| Decode slow | Matmul kernels, attention implementation |
| High variance | Background processes, thermal throttling |
| Memory regression | Buffer pool usage, tensor allocation |

## Key Grep Patterns

| Pattern | Purpose |
|---------|---------|
| `"tok/s\|tokens_per"` | Find throughput metrics |
| `"TTFT\|ttft"` | Find latency metrics |
| `"memory\|Memory"` | Find memory usage |
| `"regression\|slower"` | Find comparison results |

## Reference Files

For detailed information, consult these files:

- **Benchmark harness**: `tests/benchmark/pipeline-benchmark.ts`
- **Result format**: `docs/design/BENCHMARK_HARNESS.md`
- **Historical results**: `tests/results/*.json`
- **Regression guide**: `docs/PERFORMANCE_REGRESSION_INVESTIGATION.md`

## Related Skills

- Use `doppler-debug` if benchmark fails or produces errors
- Use `doppler-convert` to test different quantization levels
