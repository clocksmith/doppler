---
name: doppler-debug
description: Diagnose inference regressions with Doppler's shared browser/Node command contract, runtime presets, and report artifacts. (project)
---

# DOPPLER Debug Skill

Use this skill when generation fails, outputs drift, or Node/browser parity breaks.
Use this skill with `doppler-bench` when investigating performance regressions.

## Mandatory Style Guides

Read these before non-trivial debug-flow, parity, or harness-contract changes:
- `docs/style/general-style-guide.md`
- `docs/style/javascript-style-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/command-interface-design-guide.md`
- `docs/style/harness-style-guide.md`

## Developer Guide Routing

When debugging turns into extension work, also open:
- `docs/developer-guides/README.md`

Common routes:
- new manifest/runtime knob or missing contract field: `docs/developer-guides/07-manifest-runtime-field.md`
- command-surface parity fix that becomes a new command or contract surface: `docs/developer-guides/12-command-surface.md`
- missing kernel or runtime path: `docs/developer-guides/11-wgsl-kernel.md`
- attention-path bug that requires a new implementation: `docs/developer-guides/13-attention-variant.md`
- cache/layout bug that requires a new layout strategy: `docs/developer-guides/15-kvcache-layout.md`
- family-level onboarding gap: `docs/developer-guides/composite-model-family.md` or `docs/developer-guides/composite-pipeline-family.md`

## Execution Plane Contract

- Contract and tunables are declared in JSON (`runtime` + harness contract); do not substitute behavior in-place.
- JS coordinates deterministic execution: resolve/validate config, dispatch pipelines, and collect artifacts.
- WGSL executes resolved kernels; any policy branch belongs to config/rule selection before dispatch.
- For parity checks, command intent must match: unknown/mismatched intent is a failure, not an alternate path.

## Required Debug Ladder

Use this order for inference failures that load successfully but generate bad output:

1. Classify the failure:
- tokenization/chat-template
- conversion/artifact integrity
- runtime numerics
- surface/harness parity
- benchmark-only

2. Establish a trusted reference before patching runtime code:
- exact prompt text
- exact token IDs
- one early activation slice
- one output/logits slice

3. Compare boundary-by-boundary and stop at first divergence:
- embeddings
- post input norm
- Q/K/V pre-RoPE
- Q/K post-RoPE
- attention output
- FFN output
- final logits

4. Once token IDs or embeddings match, stop changing prompt wrappers or harness formatting until later evidence requires it.

5. For quantized failures, run one F16 or source-precision control before changing quantized kernels.
- F16/source-precision good + quantized bad => quantized path issue
- F16/source-precision bad + quantized bad => shared conversion/layout/runtime issue

6. Prefer one config-driven probe over one new theory.

Reference workflow: `docs/debug-playbook.md`
Reusable report template: `docs/debug-investigation-template.md`

## Conversion Completion Discipline

Do not report conversion success unless all of these are true:
- process exited successfully
- `manifest.json` exists
- shard set is present
- conversion report is valid

If shards exist without a manifest, classify the output as interrupted/incomplete and clean or overwrite it before retrying. Do not treat it as a reference artifact.

## Fast Triage

```bash
# Primary debug run (auto surface = node-first transport; browser fallback only when node transport is unavailable)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}' --json

# Verify pass/fail with inference suite
npm run verify:model -- --config '{"request":{"workload":"inference","modelId":"MODEL_ID","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}' --json

# Force browser relay for mobile/WebGPU parity checks
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"diagnostics/debug-logits"},"run":{"surface":"browser","browser":{"channel":"chrome","console":true}}}' --json
```

## Runtime Overrides (Config-First)

Use runtime JSON patches instead of ad-hoc flags:

```bash
npm run debug -- \
  --config '{"request":{"modelId":"MODEL_ID"},"run":{"surface":"auto"}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"investigate"},"debug":{"trace":{"enabled":true,"categories":["attn","ffn"],"maxDecodeSteps":2}}},"inference":{"batching":{"maxTokens":8},"sampling":{"temperature":0}}}' \
  --json
```

## Perf-Focused Investigation

```bash
# Investigate-mode profile run (trace/profiler enabled by preset)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-profile"},"run":{"surface":"auto"}}' --json

# Fast readback sensitivity checks
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r1","cacheMode":"warm"},"run":{"surface":"browser"}}' --json
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"experiments/gemma3-investigate-readback-r8","cacheMode":"warm"},"run":{"surface":"browser"}}' --json

# Direct override for decode cadence tuning
npm run bench -- \
  --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser"}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"investigate"}},"inference":{"batching":{"batchSize":4,"readbackInterval":4,"stopCheckMode":"per-token","maxTokens":128},"sampling":{"temperature":0}}}' \
  --json
```

Notes:
- `runtime.shared.tooling.intent="calibrate"` forbids trace/profiler instrumentation.
- Set `runtime.shared.tooling.intent="investigate"` for profiling/tracing runs.

## Cache and Surface Control

```bash
# Cold browser run (wipe OPFS cache before launch)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","cacheMode":"cold"},"run":{"surface":"browser"}}' --json

# Warm browser run (reuse OPFS cache)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser"}}' --json
```

## What to Inspect in Results

- `result.metrics.modelLoadMs`, `result.metrics.firstTokenMs`
- `result.metrics.prefillTokensPerSecTtft` (preferred) and `result.metrics.prefillTokensPerSec`
- `result.metrics.decodeTokensPerSec`
- `result.metrics.gpu` (if available)
- `result.memoryStats`
- `result.deviceInfo`
- `result.reportInfo` (report backend/path)

## Canonical Files

- `tools/doppler-cli.js`
- `src/tooling/command-api.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-browser-command-runner.js`
- `src/inference/browser-harness.js`
- `src/config/presets/runtime/profiles/verbose-trace.json`
- `docs/developer-guides/README.md`

## Related Skills

- `doppler-bench` for perf regression quantification
- `doppler-convert` when conversion integrity is suspected
