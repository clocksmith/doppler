---
name: doppler-perf-squeeze
description: Diagnose and improve Doppler decode/prefill performance with parity baselines, profiling traces, and controlled runtime/code experiments. (project)
---

# DOPPLER Perf Squeeze Skill

Use this skill when Doppler is slower than expected on decode, prefill, TTFT, or model-load-sensitive warm UX.

## Mandatory Style Guides

Read these before non-trivial performance, profiling, or methodology changes:
- `docs/style/general-style-guide.md`
- `docs/style/javascript-style-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/harness-style-guide.md`
- `docs/style/benchmark-style-guide.md`

Also read:
- `docs/style/wgsl-style-guide.md` for shader changes
- `docs/style/command-interface-design-guide.md` when changing `bench` or `debug` command behavior

## Developer Guide Routing

When performance work requires additive implementation changes, also open:
- `docs/developer-guides/README.md`

Common routes:
- execution-plan or kernel-path tuning: `docs/developer-guides/06-kernel-path-config.md`
- new or revised kernel implementations: `docs/developer-guides/11-wgsl-kernel.md`
- attention hot-path redesign: `docs/developer-guides/13-attention-variant.md`
- cache/layout redesign: `docs/developer-guides/15-kvcache-layout.md`
- benchmark or command contract additions: `docs/developer-guides/12-command-surface.md`

## Execution Plane Contract

- The benchmarking and tuning contract is JSON-first (`runtime` profiles + workload contracts).
- JS orchestrates timing, profiling hooks, and runtime overlays without policy drift from command examples.
- WGSL changes are code-path changes only; throughput/quality conclusions must be tied to explicit config deltas.
- Any implicit fallback (e.g., automatic policy downgrades) invalidates baseline comparisons and should be blocked.

## Workflow

### 1) Establish Baselines (Parity First)

```bash
# Fair compute comparison (Doppler cadence aligned to TJS-style per-token cadence)
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json

# Throughput-oriented Doppler cadence probe
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# Warm UX comparison (includes model load and first response path)
node tools/compare-engines.js --mode warm --warmup 1 --runs 3 --decode-profile parity --save --json
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
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r1","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json

# Deferred readback profile (r8)
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r8","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json

# Explicit custom cadence override
npm run bench -- \
  --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"investigate"}},"inference":{"batching":{"batchSize":4,"readbackInterval":4,"stopCheckMode":"per-token","maxTokens":128},"sampling":{"temperature":0}}}' \
  --json
```

### 3) Run Profiling/Tracing for Bottlenecks

```bash
# Profiling profile (investigate intent + profiler enabled)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-profile"},"run":{"surface":"auto"}}' --json
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
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile throughput --save --json

# Keep vendor registry coverage valid
node tools/vendor-bench.js validate
node tools/vendor-bench.js capabilities
node tools/vendor-bench.js gap --base doppler --target transformersjs
```

## Reporting Template

For each perf iteration, capture:
- `baseline` command + result file
- `change` (runtime-only or code patch)
- `after` command + result file
- metric deltas for `model load`, `decode tok/s`, `prompt tok/s (TTFT)`, `TTFT`
- regression risk notes (correctness, determinism, memory)

## Related Skills

- `doppler-bench` for benchmark execution and vendor normalization.
- `doppler-debug` for correctness checks while tuning performance.
- `doppler-kernel-reviewer` for WGSL/JS kernel quality review on perf patches.
