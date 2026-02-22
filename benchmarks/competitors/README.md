# Competitor Benchmark Registry

This folder is the source of truth for cross-product benchmark comparisons.
It is intentionally separate from style guides and from Doppler-only benchmark notes.

## Purpose

- Track competitor targets in one machine-readable registry.
- Keep one harness definition per competitor.
- Normalize external benchmark outputs into a shared comparison record.
- Gate benchmark claims through reproducible CI checks and normalized result artifacts.

## Registry Files

- `registry.json`: canonical list of competitor products and harness links.
- `workloads.json`: shared workload IDs used for apples-to-apples comparisons.
  - includes `defaults.compareEngines`, used by `tools/compare-engines.js` when no explicit workload/prompt/token lengths are passed.
- `capabilities.json`: capability matrix for bench/profiling coverage by target.
- `harnesses/*.json`: one harness definition per competitor.
- `schema/*.json`: schemas for registry, harness, capabilities, and normalized result records.
- `results/`: normalized comparison outputs.

## Closed Workstream Snapshot (2026-02-22 UTC)

- Gemma 3 Q4K `f32a` now auto-selects `gemma3-q4k-dequant-f32a-online` on subgroup-capable devices (`src/rules/inference/kernel-path.rules.json`).
- Kernel path registry marks `gemma3-q4k-dequant-f32a-online` as canonical/default (`src/config/presets/kernel-paths/registry.json`).
- Structural CI sweep for Gemma 3 1b kernel-path invariants is enforced by `tests/inference/gemma3-1b-kernel-sweep.test.js`.
- Inference guard workflow now triggers on inference rule changes and executes the sweep gate (`.github/workflows/inference-guard.yml`).
- Historical local performance numbers are stale after kernel-path routing updates; re-run apples-to-apples benchmark suites before publishing comparative claims.

Execution tracking now lives in:

- normalized artifacts under `benchmarks/competitors/results/`
- harness + workload contracts in this folder
- CI gates in `.github/workflows/inference-guard.yml`

## CLI

Use `tools/competitor-bench.js`:

- `node tools/competitor-bench.js list`
- `node tools/competitor-bench.js validate`
- `node tools/competitor-bench.js capabilities`
- `node tools/competitor-bench.js capabilities --target transformersjs`
- `node tools/competitor-bench.js gap --base doppler --target transformersjs`
- `node tools/competitor-bench.js show --target webllm`
- `node tools/competitor-bench.js import --target webllm --input /tmp/webllm-result.json`
- `node tools/competitor-bench.js run --target webllm --workload decode-64-128-greedy -- node ./path/to/runner.js`

`import` and `run` both produce normalized records under `benchmarks/competitors/results/` unless `--output` is specified.

## Normalization Notes

- Prefill throughput comparisons are normalized as `prompt_tokens / ttft_ms`.
- Prefer explicit source metric keys like `prefill_tokens_per_sec_ttft` when available, with legacy key fallback for older harness outputs.
- `tools/compare-engines.js` defaults to `--decode-profile parity` (Doppler `batchSize=1`, `readbackInterval=1`) for closer Transformers.js decode cadence matching; use `--decode-profile throughput` for Doppler-tuned runs.
