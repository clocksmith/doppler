---
name: doppler-debug
description: Diagnose inference regressions with Doppler's shared browser/Node command contract, runtime presets, and report artifacts. (project)
---

# DOPPLER Debug Skill

Use this skill when generation fails, outputs drift, or Node/browser parity breaks.
Use this skill with `doppler-bench` when investigating performance regressions.

## Fast Triage

```bash
# Primary debug run (auto surface: Node first, browser fallback)
npm run debug -- --model-id MODEL_ID --runtime-preset modes/debug --surface auto --json

# Verify pass/fail with inference suite
npm run test:model -- --suite inference --model-id MODEL_ID --runtime-preset modes/debug --surface auto --json

# Force browser relay for mobile/WebGPU parity checks
npm run debug -- --model-id MODEL_ID --runtime-preset diagnostics/debug-logits --surface browser --browser-channel chrome --browser-console --json
```

## Runtime Overrides (Config-First)

Use runtime JSON patches instead of ad-hoc flags:

```bash
npm run debug -- --model-id MODEL_ID --surface auto --runtime-config-json '{"shared":{"tooling":{"intent":"investigate"},"debug":{"trace":{"enabled":true,"categories":["attn","ffn"],"maxDecodeSteps":2}}},"inference":{"batching":{"maxTokens":8},"sampling":{"temperature":0}}}' --json
```

## Perf-Focused Investigation

```bash
# Investigate-mode profile run (trace/profiler enabled by preset)
npm run debug -- --model-id MODEL_ID --runtime-preset experiments/gemma3-profile --json

# Fast readback sensitivity checks
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-investigate-readback-r1 --cache-mode warm --json
npm run bench -- --model-id MODEL_ID --runtime-preset experiments/gemma3-investigate-readback-r8 --cache-mode warm --json

# Direct override for decode cadence tuning
npm run bench -- --model-id MODEL_ID --cache-mode warm --runtime-config-json '{"shared":{"tooling":{"intent":"investigate"}},"inference":{"batching":{"batchSize":4,"readbackInterval":4,"stopCheckMode":"per-token","maxTokens":128},"sampling":{"temperature":0}}}' --json
```

Notes:
- `runtime.shared.tooling.intent="calibrate"` forbids trace/profiler instrumentation.
- Set `runtime.shared.tooling.intent="investigate"` for profiling/tracing runs.

## Cache and Surface Control

```bash
# Cold browser run (wipe OPFS cache before launch)
npm run debug -- --model-id MODEL_ID --surface browser --cache-mode cold --json

# Warm browser run (reuse OPFS cache)
npm run debug -- --model-id MODEL_ID --surface browser --cache-mode warm --json
```

## What to Inspect in Results

- `result.metrics.modelLoadMs`, `result.metrics.ttftMs`
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
- `src/config/presets/runtime/modes/debug.json`

## Related Skills

- `doppler-bench` for perf regression quantification
- `doppler-convert` when conversion integrity is suspected
