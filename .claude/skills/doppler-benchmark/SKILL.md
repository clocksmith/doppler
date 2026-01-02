---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks. Use when measuring inference speed, comparing against baselines, or tracking performance regressions. Outputs JSON results per the BENCHMARK_HARNESS spec. (project)
---

# DOPPLER Benchmark

## Using Config Presets

```bash
# Use built-in bench preset (silent output, deterministic sampling)
npm run bench -- --config bench -m MODEL

# Use CI preset (file logging, short timeout)
npm run bench -- --config ci -m MODEL

# List available presets
npx tsx cli/index.ts --list-presets
```

Note: `--config` loads runtime presets only. Model presets are separate.

## Fast Iteration (use --skip-load after first run!)

```bash
# First run - loads model (~30s), keeps browser open
npm run bench -- --config bench -m MODEL --warm 2>&1 | grep --line-buffered -E "tok/s|TTFT|Done" | sed '/Done/q'

# Subsequent runs - reuses model in GPU RAM
npm run bench -- --config bench -m MODEL --skip-load 2>&1 | grep --line-buffered -E "tok/s|TTFT|Done" | sed '/Done/q'

# Multiple runs for statistics
npm run bench -- --config bench -m MODEL --skip-load --runs 3

# Save and compare
npm run bench -- --config bench -m MODEL -o baseline.json
npm run bench -- --config bench -m MODEL --compare baseline.json
```

## Key: `sed '/Done/q'` exits after Done line

## After Code Changes

```bash
npm run build && npm run bench -- --config bench -m MODEL --skip-load 2>&1 | grep --line-buffered -E "tok/s|Done" | sed '/Done/q'
```

## Key Metrics

- `decode_tokens_per_sec` - Main throughput metric
- `ttft_ms` - Time to first token
- `prefill_tokens_per_sec` - Prefill speed

## CI Benchmarking

Use the CI preset with file logging:

```bash
npm run bench -- --config ci -m MODEL -o results.json
```

Or create a custom CI config:

```json
{
  "extends": "bench",
  "runtime": {
    "debug": {
      "logOutput": { "stdout": false, "file": "./ci-logs/bench.log" }
    }
  },
  "cli": { "timeout": 180000 }
}
```
