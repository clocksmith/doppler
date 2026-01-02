---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks. Use when measuring inference speed, comparing against baselines, or tracking performance regressions. Outputs JSON results per the BENCHMARK_HARNESS spec. (project)
---

# Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `decode_tokens_per_sec` | Main throughput metric | Higher is better |
| `ttft_ms` | Time to first token (latency) | Lower is better |
| `prefill_tokens_per_sec` | Prompt processing speed | Context-dependent |
| `p99_latency_ms` | Tail latency | Flag if >100ms |

# Standard Commands

```bash
# Quick benchmark - exits after completion
npm run bench -- -m MODEL 2>&1 | grep -E "tok/s|TTFT|Done"

# Multiple runs for statistical confidence
npm run bench -- -m MODEL --runs 3 2>&1 | sed '/Done/q'

# Save baseline for comparison
npm run bench -- -m MODEL -o baseline.json
npm run bench -- -m MODEL --compare baseline.json
```

# Fast Iteration

```bash
# First run loads model, subsequent runs reuse GPU memory
npm run bench -- -m MODEL --skip-load 2>&1 | sed '/Done/q'

# After code changes - rebuild then benchmark
npm run build && npm run bench -- -m MODEL --skip-load 2>&1 | sed '/Done/q'
```

# Regression Detection

When asked "Is this slower?", always:

1. Run benchmark on current code with `--runs 3`
2. Compare against saved baseline JSON if available
3. Report decode_tokens_per_sec difference as percentage
4. Flag regressions >5% as significant

# Interpretation Guidelines

- **Prefill vs Decode**: Prefill is memory-bound, decode is compute-bound
- **Batch size 1**: Standard for interactive inference
- **Cold vs Warm**: First run includes model loading, use `--skip-load` for warm

# Key Grep Patterns

`"tok/s|TTFT|latency"` - Performance metrics. `"regression|slower"` - Problems.
