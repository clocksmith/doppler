---
name: doppler-perf-squeeze
description: Diagnose and improve Doppler decode/prefill performance with parity baselines, profiling traces, and controlled runtime/code experiments. (project)
---

# DOPPLER Perf Squeeze Skill

Use this skill when Doppler is slower than expected on decode, prefill, TTFT, or model-load-sensitive warm UX.

## Workflow

### 1) Establish Baselines (Parity First)

```bash
# Fair compute comparison (Doppler cadence aligned to TJS-style per-token cadence)
node tools/compare-engines.mjs --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json

# Throughput-oriented Doppler cadence probe
node tools/compare-engines.mjs --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# Warm UX comparison (includes model load and first response path)
node tools/compare-engines.mjs --mode warm --warmup 1 --runs 3 --decode-profile parity --save --json
```

Read from output:
- `model load`
- `decode tok/s`
- `prompt tok/s (TTFT)`
- `TTFT`

Methodology note:
- Compare harness normalizes prefill as `prompt_tokens / ttft_ms`.

### 2) Sweep Runtime Decode Cadence

```bash
# Readback-heavy profile (r1)
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-investigate-readback-r1 --cache-mode warm --save --json

# Deferred readback profile (r8)
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-investigate-readback-r8 --cache-mode warm --save --json

# Explicit custom cadence override
npm run bench -- --model-id MODEL_ID --cache-mode warm --runtime-config-json '{"shared":{"tooling":{"intent":"investigate"}},"inference":{"batching":{"batchSize":4,"readbackInterval":4,"stopCheckMode":"per-token","maxTokens":128},"sampling":{"temperature":0}}}' --save --json
```

### 3) Run Profiling/Tracing for Bottlenecks

```bash
# Profiling preset (investigate intent + profiler enabled)
npm run debug -- --model-id MODEL_ID --runtime-preset experiments/gemma3-profile --json
```

Rules:
- Keep `runtime.shared.tooling.intent="investigate"` for profile/trace runs.
- `bench` enforces `intent="calibrate"` and will reject profiler/trace instrumentation.
- Do not compare investigate-mode numbers against calibrate-mode baselines without labeling them.

### 4) Form Hypotheses and Patch

Common patterns:
- Decode slowdown:
  - Too-frequent readback/map on logits path.
  - Excess submit/sync boundaries per token.
- Prefill slowdown:
  - Full-sequence logits readback when only last-position logits are needed.
  - Extra tensor materialization in prefill path.

Priority code hotspots:
- `src/inference/pipelines/text/logits/index.js`
- `src/inference/pipelines/text/generator.js`
- `src/inference/browser-harness.js`
- `src/memory/buffer-pool.js`

### 5) Re-Measure and Gate

```bash
# Re-run parity and throughput comparisons after each material change
node tools/compare-engines.mjs --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json
node tools/compare-engines.mjs --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# Keep competitor registry coverage valid
node tools/competitor-bench.js validate
node tools/competitor-bench.js capabilities
node tools/competitor-bench.js gap --base doppler --target transformersjs
```

## Reporting Template

For each perf iteration, capture:
- `baseline` command + result file
- `change` (runtime-only or code patch)
- `after` command + result file
- metric deltas for `model load`, `decode tok/s`, `prompt tok/s (TTFT)`, `TTFT`
- regression risk notes (correctness, determinism, memory)

## Related Skills

- `doppler-bench` for benchmark execution and competitor normalization.
- `doppler-debug` for correctness checks while tuning performance.
- `doppler-kernel-reviewer` for WGSL/JS kernel quality review on perf patches.
