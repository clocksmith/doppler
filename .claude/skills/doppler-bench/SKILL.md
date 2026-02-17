---
name: doppler-bench
description: Run Doppler and competitor benchmark workflows, capture reproducible JSON artifacts, and compare bench/profile coverage using the competitor registry. (project)
---

# DOPPLER Bench Skill

Use this skill for repeatable performance measurement and cross-product comparisons.

## Doppler Benchmark (Primary)

```bash
# Warm-cache benchmark (recommended baseline)
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-bench-q4k --cache-mode warm --save --json

# Cold-cache benchmark (cache disabled per run)
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-bench-q4k --cache-mode cold --save --json

# Compare against last saved run
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-bench-q4k --compare last --save --json
```

Notes:
- `bench` defaults to browser surface and persistent Chromium profile.
- Saved artifacts go to `bench-results/` when `--save` is used.

## Competitor Benchmark (Transformers.js)

```bash
# Raw Transformers.js benchmark with ORT op profiling summary
node external/transformersjs-bench.mjs --workload decode-64-128-greedy --cache-mode warm --profile-ops on --profile-top 20 --json

# Normalize result into competitor registry output
node tools/competitor-bench.js run --target transformersjs --workload decode-64-128-greedy -- node external/transformersjs-bench.mjs --workload decode-64-128-greedy --cache-mode warm --profile-ops on --profile-top 20 --json
```

## Coverage Tracking (Bench vs Profile)

```bash
# Validate registry + harness + capability matrix
node tools/competitor-bench.js validate

# Show capability coverage for all targets
node tools/competitor-bench.js capabilities

# Show exact Doppler -> Transformers.js feature gaps
node tools/competitor-bench.js gap --base doppler --target transformersjs
```

## Key Metrics

- `decode_tokens_per_sec`
- `prefill_tokens_per_sec`
- `ttft_ms`
- `decode_ms_per_token_p50/p95`
- `model_load_ms`
- `ort_profiled_total_ms` (Transformers.js harness)

## Canonical Files

- `tools/doppler-cli.js`
- `external/transformersjs-bench.mjs`
- `external/transformersjs-runner.html`
- `benchmarks/competitors/registry.json`
- `benchmarks/competitors/capabilities.json`
- `benchmarks/competitors/results/`

## Related Skills

- `doppler-debug` for correctness regressions discovered during bench runs
- `doppler-convert` when conversion format/quantization differences affect perf
