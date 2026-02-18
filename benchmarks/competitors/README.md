# Competitor Benchmark Registry

This folder is the source of truth for cross-product benchmark comparisons.
It is intentionally separate from style guides and from Doppler-only benchmark notes.

## Purpose

- Track competitor targets in one machine-readable registry.
- Keep one harness definition per competitor.
- Normalize external benchmark outputs into a shared comparison record.
- Keep one execution-focused competitive playbook tied to benchmark evidence.

## Registry Files

- `registry.json`: canonical list of competitor products and harness links.
- `workloads.json`: shared workload IDs used for apples-to-apples comparisons.
- `capabilities.json`: capability matrix for bench/profiling coverage by target.
- `harnesses/*.json`: one harness definition per competitor.
- `schema/*.json`: schemas for registry, harness, capabilities, and normalized result records.
- `results/`: normalized comparison outputs.
- `STRATEGY.md`: competitive attack plan and KPI/roadmap targets.

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
- `tools/compare-engines.mjs` defaults to `--decode-profile parity` (Doppler `batchSize=1`, `readbackInterval=1`) for closer Transformers.js decode cadence matching; use `--decode-profile throughput` for Doppler-tuned runs.
