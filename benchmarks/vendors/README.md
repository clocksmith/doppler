# Vendor Benchmark Registry

This folder is the source of truth for cross-product benchmark comparisons.
It is intentionally separate from style guides and from Doppler-only benchmark notes.

## Purpose

- Track vendor targets in one machine-readable registry.
- Keep one harness definition per vendor.
- Normalize external benchmark outputs into a shared comparison record.
- Gate benchmark claims through reproducible CI checks and normalized result artifacts.

## How to interpret results

What these benchmarks prove:
- Phase timing split under one shared contract (`modelLoadMs`, `firstTokenMs`, `firstResponseMs`, `prefillMs`, `decodeMs`, throughput, and decode p50/p95/p99).
- Relative behavior for the same workload/sampling/cache/load settings on the same browser + machine profile.
- Whether configuration and load-path changes map to measurable differences.

What these benchmarks do not prove:
- Internet/WAN performance (local loopback/LAN load-path tests are not WAN tests).
- Cross-hardware absolute rankings.
- Quality parity beyond the explicit correctness checks captured in the run.

Claim format to keep reports auditable:
- State the workload and cache/load mode.
- State engine/version (`Doppler`, `Transformers.js (v4)`).
- Include the exact command and artifact path under `benchmarks/vendors/results/`.

## Registry Files

- `registry.json`: canonical list of vendor products and harness links.
- `workloads.json`: shared workload IDs used for apples-to-apples comparisons.
  - includes `defaults.compareEngines`, used by `tools/compare-engines.js` when no explicit workload/prompt/token lengths are passed.
- `capabilities.json`: capability matrix for bench/profiling coverage by target.
- `harnesses/*.json`: one harness definition per vendor.
- `schema/*.json`: schemas for registry, harness, capabilities, metric contract, and normalized result records.
- `schema/compare-engines-config.schema.json`: schema for `compare-engines.config.json`.
- `results/`: normalized comparison outputs.
- `compare-metrics.json`: shared compare metric contract for CLI and harness-driven extraction.

## Closed Workstream Snapshot (2026-02-22 UTC)

- Gemma 3 Q4K `f32a` now auto-selects `gemma3-q4k-dequant-f32a-online` on subgroup-capable devices (`src/rules/inference/kernel-path.rules.json`).
- Kernel path registry marks `gemma3-q4k-dequant-f32a-online` as canonical/default (`src/config/presets/kernel-paths/registry.json`).
- Structural CI sweep for Gemma 3 1b kernel-path invariants is enforced by `tests/inference/gemma3-1b-kernel-sweep.test.js`.
- Inference guard workflow now triggers on inference rule changes and executes the sweep gate (`.github/workflows/inference-guard.yml`).
- Historical local performance numbers are stale after kernel-path routing updates; re-run apples-to-apples benchmark suites before publishing comparative claims.

Execution tracking now lives in:

- normalized artifacts under `benchmarks/vendors/results/`
- harness + workload contracts in this folder
- CI gates in `.github/workflows/inference-guard.yml`

## CLI

Use `tools/vendor-bench.js`:

- `node tools/vendor-bench.js list`
- `node tools/vendor-bench.js validate`
- `node tools/vendor-bench.js capabilities`
- `node tools/vendor-bench.js capabilities --target transformersjs`
- `node tools/vendor-bench.js gap --base doppler --target transformersjs`
- `node tools/vendor-bench.js show --target webllm`
- `node tools/vendor-bench.js import --target webllm --input /tmp/webllm-result.json`
- `node tools/vendor-bench.js run --target webllm --workload decode-64-128-greedy -- node ./path/to/runner.js`

`import` and `run` both produce normalized records under `benchmarks/vendors/results/` unless `--output` is specified.

## Normalization Notes

- Canonical timing contract includes:
  - `decodeTokensPerSec`
  - `prefillTokensPerSec`
  - `firstTokenMs`
  - `firstResponseMs`
  - `prefillMs`
  - `decodeMs`
  - `totalRunMs`
  - `modelLoadMs`
  - `decodeMsPerTokenP50`
  - `decodeMsPerTokenP95`
  - `decodeMsPerTokenP99`
- `cacheMode` and `loadMode` are required under each run's `timing` object (`cacheMode`: `cold|warm`, `loadMode`: `opfs|http|memory`).
- Harness mappings are now one-path canonical only (`normalization.metricPaths` / `metadataPaths` entries are single-string arrays). Mixed fallbacks are rejected by schema.
- Metric paths are canonicalized through `benchmarks/vendors/harnesses/*.json` and validated as required before any comparison.
- `tools/compare-engines.js` defaults to `--decode-profile parity` (Doppler `batchSize=1`, `readbackInterval=1`) for closer Transformers.js decode cadence matching; use `--decode-profile throughput` for Doppler-tuned runs.

## Visualization

Use `benchmarks/vendors/compare-chart.js` to turn a saved compare result file into an SVG:

```bash
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/results/compare_latest.json
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/results/compare_latest.json --chart stacked
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/results/compare_latest.json --chart radar --section compute/parity
```

Use `--section` to choose the section, `--chart` (`bar|stacked|radar`) to pick the renderer, and `--metrics` to limit metric IDs.
