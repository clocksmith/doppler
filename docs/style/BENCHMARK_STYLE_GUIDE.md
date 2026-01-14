# DOPPLER Benchmark Style Guide

Benchmarking conventions for DOPPLER. Benchmarks are test harnesses, not runtime code.

## Output Schema

- Emit JSON that conforms to `docs/spec/BENCHMARK_SCHEMA.json`.
- Always include `schemaVersion`, `timestamp`, and `suite`.
- Include `env`, `model`, `workload`, `metrics`, and `raw` when available.

## Baseline Comparison

- Use the CLI `--compare <baseline.json>` for regression checks.
- Respect `runtime.shared.benchmark.comparison.regressionThresholdPercent`.
- Fail the run when `failOnRegression` is enabled and any metric regresses beyond the threshold.

## Run Policy

- Keep warmup and timed run counts in `runtime.shared.benchmark.run`.
- Use `runtime.shared.benchmark.stats` for outlier filtering, warmup stability, and thermal detection thresholds.

## Stats

- Use `src/debug/stats.js` for percentiles, IQR outlier removal, and confidence intervals.
- Avoid duplicate stats implementations in test harnesses.

## Profiling

- Use `gpu/profiler.js` for GPU timestamps (not ad-hoc timers).
- Keep CPU timing in benchmark harnesses as a fallback.
