---
name: doppler-perf
description: Diagnose and improve Doppler model/path performance with baselines, profiling traces, and controlled runtime/code experiments. (project)
---

# DOPPLER Perf Skill

Use this skill when Doppler is slower than expected on decode, prefill, TTFT, or model-load-sensitive warm UX, and you need to diagnose or change the hot path for a specific model or runtime path. Use `doppler-bench` when the goal is reproducible benchmark evidence, compare-engine reporting, or vendor-registry coverage rather than tuning.

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

### 1) Establish Baselines

```bash
# Start from one clean benchmark baseline
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"profiles/throughput","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json
```

Read from output:
- `model load`
- `decode tok/s`
- `prompt tok/s (TTFT)`
- `TTFT`

If you need compare-engine or publication-grade evidence at this stage, switch to `doppler-bench` instead of expanding the squeeze loop.

### 2) Localize The Phase And The Wall

Before changing kernels:
- Treat prefill and decode as separate investigations.
- Dump resolved prefill/decode kernel path and the actual loaded weight/materialization dtype for the hot ops.
- For decode, record `decodeMode`, `batchGuardReason`, speculation state, `decodeRecordMs`, `decodeSubmitWaitMs`, `decodeReadbackWaitMs`, `singleTokenReadbackWaitMs`, and `singleTokenOrchestrationMs`.
- If submit/readback dominates, fix orchestration first; do not assume the math kernel is the main wall.

### 3) Sweep Runtime Decode Cadence

```bash
# Single-token-style control
npm run bench -- \
  --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":128},"sampling":{"temperature":0,"topK":1,"topP":1,"repetitionPenalty":1,"greedyThreshold":0},"session":{"decodeLoop":{"batchSize":1,"stopCheckMode":"batch","readbackInterval":1,"ringTokens":1,"ringStop":1,"ringStaging":1,"disableCommandBatching":false}}}}' \
  --json

# Moderate batched candidate
npm run bench -- \
  --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":128},"sampling":{"temperature":0,"topK":1,"topP":1,"repetitionPenalty":1,"greedyThreshold":0},"session":{"decodeLoop":{"batchSize":4,"stopCheckMode":"batch","readbackInterval":4,"ringTokens":1,"ringStop":1,"ringStaging":1,"disableCommandBatching":false}}}}' \
  --json

# Higher-throughput candidate
npm run bench -- \
  --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":128},"sampling":{"temperature":0,"topK":1,"topP":1,"repetitionPenalty":1,"greedyThreshold":0},"session":{"decodeLoop":{"batchSize":8,"stopCheckMode":"batch","readbackInterval":8,"ringTokens":1,"ringStop":1,"ringStaging":1,"disableCommandBatching":false}}}}' \
  --json
```

### 3) Run Profiling/Tracing for Bottlenecks

```bash
# Trace-heavy debug run
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}' --json

# Logit-focused browser investigation
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"diagnostics/debug-logits"},"run":{"surface":"browser","browser":{"channel":"chrome","console":true}}}' --json
```

Rules:
- `bench` is calibrate-only; do not override its intent in runtime config.
- `debug` is the investigate surface for traces, layer probes, and diagnostics.
- Do not compare investigate-mode numbers against calibrate-mode baselines without labeling them.

### 4) Form Hypotheses and Patch

**Correctness gate:** If a performance change (kernel swap, transform, materialization path) produces incorrect output — garbled text, numerical divergence, repeated tokens — stop the perf workflow and invoke `doppler-debug` immediately. Do not continue tuning a path that produces wrong answers. The debug ladder must confirm correctness before perf measurement resumes.

**Context budget:** Investigation and diagnosis should consume no more than 30% of available context. If you have not formed a testable hypothesis after reading 5–8 files, stop reading code and run a diagnostic probe or write a minimal reproduction instead. Exhaustive static code tracing without running a test is an anti-pattern.

Common patterns:
- Decode slowdown:
  - Too-frequent readback/map on logits path.
  - Excess submit/sync boundaries per token.
- Prefill slowdown:
  - Full-sequence logits readback when only last-position logits are needed.
  - Extra tensor materialization in prefill path.
- Correctness regression from perf change:
  - Precision loss in dtype materialization path (e.g., f16 round-trip in weight dequant).
  - Transform remapping that changes kernel precision semantics (e.g., fused Q4K → GEMV with f16 weights).
  - Always measure per-op numerical error before attributing to a specific kernel.

Priority code hotspots:
- `src/inference/pipelines/text/logits/index.js`
- `src/inference/pipelines/text/generator.js`
- `src/inference/browser-harness.js`
- `src/memory/buffer-pool.js`

### 5) Re-Measure and Gate

```bash
# Re-run the clean benchmark baseline after each material change
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"profiles/throughput","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --json
```

When the final evidence will be published, compared across engines, or used to update vendor-facing numbers, hand off to `doppler-bench` instead of extending the squeeze loop.

## Reporting Template

For each perf iteration, capture:
- `baseline` command + result file
- `change` (runtime-only or code patch)
- `after` command + result file
- metric deltas for `model load`, `decode tok/s`, `prompt tok/s (TTFT)`, `TTFT`
- regression risk notes (correctness, determinism, memory)

## Protocol References

- `docs/agents/benchmark-protocol.md` — vendor benchmark registry and update checklist
- `docs/agents/hardware-notes.md` — GPU memory assumptions

## Related Skills

- `doppler-bench` for publication-grade benchmark execution, compare-engine evidence, and vendor normalization.
- `doppler-debug` for correctness checks while tuning performance.
- `doppler-kernel-reviewer` for WGSL/JS kernel quality review on perf patches.
