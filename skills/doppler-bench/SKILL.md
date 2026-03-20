---
name: doppler-bench
description: Run Doppler and vendor benchmark workflows, capture reproducible JSON artifacts, and compare bench/profile coverage using the vendor registry. (project)
---

# DOPPLER Bench Skill

Use this skill for repeatable performance measurement and cross-product comparisons.

## Mandatory Style Guides

Read these before non-trivial edits or benchmark-methodology changes:
- `docs/style/general-style-guide.md`
- `docs/style/javascript-style-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/command-interface-design-guide.md`
- `docs/style/harness-style-guide.md`
- `docs/style/benchmark-style-guide.md`

## Developer Guide Routing

When benchmark work becomes extension work, also open:
- `docs/developer-guides/README.md`

Common routes:
- tuning or adding execution identities: `docs/developer-guides/06-kernel-path-config.md`
- kernel-level perf changes: `docs/developer-guides/11-wgsl-kernel.md`
- attention-path throughput work: `docs/developer-guides/13-attention-variant.md`
- cache/layout throughput work: `docs/developer-guides/15-kvcache-layout.md`
- command-surface or harness-contract additions: `docs/developer-guides/12-command-surface.md`

## Execution Plane Contract

- JSON governs benchmark policy and engine selection (`runtimeConfig`, runtime profiles, rule assets).
- JS wraps execution: contract validation, harness/runtime assembly, config isolation, and dispatch orchestration.
- WGSL remains deterministic compute; it must not own benchmark semantics or fallback logic.
- Any benchmark fairness axis (`sampling`, `seed`, budget, run policy) must come from shared contract JSON and be identical across engines.
- Any unrepresented behavior choice must fail fast instead of falling back.

## Cross-Engine Compare (Canonical)

```bash
# Fair compute comparison (default parity decode cadence)
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json

# Doppler throughput-tuned decode cadence
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# Warm-start only (includes model load)
node tools/compare-engines.js --mode warm --warmup 1 --runs 3 --decode-profile parity --save --json
```

Notes:
- `--decode-profile parity` maps Doppler to `batchSize=1`, `readbackInterval=1` for closer TJS cadence parity.
- `--decode-profile throughput` maps Doppler to `batchSize=4`, `readbackInterval=4`.
- Prefill is normalized as `prompt_tokens / ttft_ms` in compare output.

## Doppler Benchmark (Primary)

```bash
# Warm-cache benchmark (recommended baseline)
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/bench/gemma3-bench-q4k","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json

# Cold-cache benchmark (cache disabled per run)
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/bench/gemma3-bench-q4k","cacheMode":"cold"},"run":{"surface":"browser","bench":{"save":true}}}' --json

# Compare against last saved run
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/bench/gemma3-bench-q4k","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true,"compare":"last"}}}' --json
```

Notes:
- `bench` defaults to `--surface auto`; set `run.surface="browser"` when you explicitly want the browser relay.
- Saved artifacts go to `benchmarks/vendors/results/` when `--save` is used.
- For instrumentation-heavy investigation, run `debug` with `request.runtimeProfile="experiments/gemma3-profile"`.

## Performance Investigation Loop (Squeeze Workflow)

```bash
# 1) Baseline parity
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json

# 2) Throughput probe
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# 3) Readback sensitivity (fixed workload, warm cache)
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r1","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r8","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json

# 4) Profile traces (investigate intent + profiler)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-profile"},"run":{"surface":"auto"}}' --json
```

## Vendor Benchmark (Transformers.js)

```bash
# Raw Transformers.js benchmark with ORT op profiling summary
node benchmarks/runners/transformersjs-bench.js --workload g3-p064-d064-t0-k1 --cache-mode warm --profile-ops on --profile-top 20 --json

# Normalize result into vendor registry output
node tools/vendor-bench.js run --target transformersjs --workload g3-p064-d064-t0-k1 -- node benchmarks/runners/transformersjs-bench.js --workload g3-p064-d064-t0-k1 --cache-mode warm --profile-ops on --profile-top 20 --json
```

## Coverage Tracking (Bench vs Profile)

```bash
# Validate registry + harness + capability matrix
node tools/vendor-bench.js validate

# Show capability coverage for all targets
node tools/vendor-bench.js capabilities

# Show exact Doppler -> Transformers.js feature gaps
node tools/vendor-bench.js gap --base doppler --target transformersjs
```

## Key Metrics

- `decode_tokens_per_sec`
- `prefill_tokens_per_sec_ttft` (preferred normalized prefill metric)
- `prefill_tokens_per_sec` (legacy alias)
- `ttft_ms`
- `decode_ms_per_token_p50/p95`
- `model_load_ms`
- `ort_profiled_total_ms` (Transformers.js harness)

## Canonical Files

- `src/cli/doppler-cli.js`
- `benchmarks/runners/transformersjs-bench.js`
- `benchmarks/runners/transformersjs-runner.html`
- `benchmarks/vendors/registry.json`
- `benchmarks/vendors/capabilities.json`
- `benchmarks/vendors/results/`
- `docs/developer-guides/README.md`

## Related Skills

- `doppler-debug` for correctness regressions discovered during bench runs
- `doppler-convert` when conversion format/quantization differences affect perf
