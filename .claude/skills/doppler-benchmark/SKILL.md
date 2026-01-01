---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks. Use when measuring inference speed, comparing against baselines, or tracking performance regressions. Outputs JSON results per the BENCHMARK_HARNESS spec. (project)
---

# DOPPLER Benchmark

## Fast Iteration (use --skip-load after first run!)

```bash
# First run - loads model (~30s), keeps browser open
npm run bench -- -m MODEL --warm 2>&1 | grep --line-buffered -E "tok/s|TTFT|Done" | sed '/Done/q'

# Subsequent runs - reuses model in GPU RAM
npm run bench -- -m MODEL --skip-load 2>&1 | grep --line-buffered -E "tok/s|TTFT|Done" | sed '/Done/q'

# Multiple runs for statistics
npm run bench -- -m MODEL --skip-load --runs 3

# Save and compare
npm run bench -- -m MODEL -o baseline.json
npm run bench -- -m MODEL --compare baseline.json
```

## Key: `sed '/Done/q'` exits after Done line

## After Code Changes

```bash
npm run build && npm run bench -- -m MODEL --skip-load 2>&1 | grep --line-buffered -E "tok/s|Done" | sed '/Done/q'
```

## Key Metrics

- `decode_tokens_per_sec` - Main throughput metric
- `ttft_ms` - Time to first token
- `prefill_tokens_per_sec` - Prefill speed
