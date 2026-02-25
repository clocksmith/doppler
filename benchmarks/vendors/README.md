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
- Include the exact command plus artifact paths:
  - local/generated JSON under `benchmarks/vendors/results/` (gitignored)
  - committed fixture evidence under `benchmarks/vendors/fixtures/` (when publishing claims)

## Registry Files

- `registry.json`: canonical list of vendor products and harness links.
- `workloads.json`: shared workload IDs used for apples-to-apples comparisons.
  - includes `defaults.compareEngines`, used by `tools/compare-engines.js` when no explicit workload/prompt/token lengths are passed.
- `capabilities.json`: capability matrix for bench/profiling coverage by target.
- `harnesses/*.json`: one harness definition per vendor.
- `schema/*.json`: schemas for registry, workloads, harness, capabilities, metric contract, and normalized result records.
- `schema/compare-engines-config.schema.json`: schema for `compare-engines.config.json`.
- `schema/release-matrix.schema.json`: schema for generated release/support matrix payload.
- `results/`: local/generated normalized outputs (`*.json`, gitignored) and committed chart snapshots (`*.svg`).
- `fixtures/`: committed sample compare payloads for clean-checkout chart/matrix smoke checks.
- `compare-metrics.json`: shared compare metric contract for CLI and harness-driven extraction.
- `release-matrix.json`: generated release/support matrix from registry + workloads + capabilities + model catalog (+ optional latest compare JSON).

## Closed Workstream Snapshot (2026-02-22 UTC)

- Gemma 3 Q4K `f32a` now auto-selects `gemma3-q4k-dequant-f32a-online` on subgroup-capable devices (`src/rules/inference/kernel-path.rules.json`).
- Kernel path registry marks `gemma3-q4k-dequant-f32a-online` as canonical/default (`src/config/presets/kernel-paths/registry.json`).
- Structural CI sweep for Gemma 3 1b kernel-path invariants is enforced by `tests/inference/gemma3-1b-kernel-sweep.test.js`.
- Inference guard workflow now triggers on inference rule changes and executes the sweep gate (`.github/workflows/inference-guard.yml`).
- Historical local performance numbers are stale after kernel-path routing updates; re-run apples-to-apples benchmark suites before publishing comparative claims.

Execution tracking now lives in:

- local/generated normalized artifacts under `benchmarks/vendors/results/`
- committed compare fixtures under `benchmarks/vendors/fixtures/`
- harness + workload contracts in this folder
- CI gates in `.github/workflows/inference-guard.yml`

## CLI

Use `tools/vendor-bench.js`:

- `node tools/vendor-bench.js list`
- `node tools/vendor-bench.js validate`
- `node tools/vendor-bench.js capabilities`
- `node tools/vendor-bench.js capabilities --target transformersjs`
- `node tools/vendor-bench.js gap --base doppler --target transformersjs`
- `node tools/vendor-bench.js matrix`
- `node tools/vendor-bench.js matrix --compare-result benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json`
- `node tools/vendor-bench.js matrix --include-local-results`
- `node tools/vendor-bench.js show --target webllm`
- `node tools/vendor-bench.js import --target webllm --input /tmp/webllm-result.json`
- `node tools/vendor-bench.js run --target webllm --workload decode-64-128-greedy -- node ./path/to/runner.js`

`import` and `run` both produce normalized records under `benchmarks/vendors/results/` unless `--output` is specified.
`matrix` writes `benchmarks/vendors/release-matrix.json` and `docs/release-matrix.md`.
By default, `matrix` auto-discovers compare JSON artifacts under `benchmarks/vendors/fixtures/` only.
Use `--include-local-results` to also scan `benchmarks/vendors/results/` (files containing `compare` in the name).
Use `--strict-compare-artifacts` to fail generation when any auto-discovered compare artifact is invalid.
Workload rows in the markdown include a `GPU/OS/Platform` column derived from each linked compare artifact's runtime environment metadata.
When `--compare-result` is provided, matrix generation also captures host/browser/GPU specs from that compare payload.
`tools/compare-engines.js --save` refreshes release-matrix artifacts automatically using fixture-only discovery by default (use `--skip-matrix-update` to opt out).

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
- Normalized result records now require a canonical `environment` block (`host`, `browser`, `gpu`, `runtime`) so platform/hardware context is always captured in benchmark JSON.
- For `vendor-bench run`, missing core environment capture fields fail normalization (`host`, browser identity, GPU identity, backend, runtime device/library).
- Harness mappings allow ordered fallback path arrays (`normalization.metricPaths` / `metadataPaths`).
- Path order is canonicalized in harness files and validated before comparison.
- Metric paths are canonicalized through `benchmarks/vendors/harnesses/*.json` and validated as required before any comparison.
- `tools/compare-engines.js` defaults to `--decode-profile parity` (Doppler `batchSize=1`, `readbackInterval=1`) for closer Transformers.js decode cadence matching; use `--decode-profile throughput` for Doppler-tuned runs.

## Visualization

Use `benchmarks/vendors/compare-chart.js` to turn a saved compare result file into an SVG:

```bash
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json --chart stacked
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json --chart radar --section compute/parity
```

Use `--section` to choose the section, `--chart` (`bar|stacked|radar`) to pick the renderer, and `--metrics` to limit metric IDs.

## Change Checklist

Add a vendor target:
- Update `benchmarks/vendors/registry.json` with the product entry and harness path.
- Add/update capability flags + evidence in `benchmarks/vendors/capabilities.json`.
- Add a harness definition in `benchmarks/vendors/harnesses/<vendor>.json`.
- Run `node tools/vendor-bench.js validate` and fix schema/shape violations.
- Update this README if workflow/coverage expectations changed.

Add a workload:
- Add the workload row in `benchmarks/vendors/workloads.json`.
- Ensure it passes `benchmarks/vendors/schema/workloads.schema.json`.
- If it should be the default, update `defaults.compareEngines`.
- Run `node tools/vendor-bench.js validate`.

Add or rename a compare metric:
- Update `benchmarks/vendors/compare-metrics.json` (id/label/unit/higherBetter/required).
- Ensure harness path mappings are present in both Doppler and Transformers.js harness files.
- Run `node tools/compare-engines.js --help` sanity checks and a sample compare run.
- Regenerate chart artifacts if metric display is expected in committed visuals.
