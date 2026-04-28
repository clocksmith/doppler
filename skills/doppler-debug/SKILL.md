---
name: doppler-debug
description: Diagnose inference regressions with Doppler's shared browser/Node command contract, runtime profiles, and report artifacts. (project)
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

## Context Budget Discipline

Investigation and diagnosis should consume no more than 30% of available context. If you have not formed a testable hypothesis after reading 5–8 files, stop reading code and do one of:
- Write a minimal reproduction script that isolates the failing path.
- Run a `doppler-debug` probe to capture per-layer numerical readback at the suspected divergence point.
- Narrow scope by diffing the execution graph before/after the suspected change.

Exhaustive static code tracing across an entire pipeline (kernel dispatch → selection → config → shader) without running a diagnostic is an anti-pattern. Each file read should either confirm or refute a specific hypothesis. If it does neither, the hypothesis is too vague — sharpen it before continuing.

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

3. Dump the phase-specific execution contract:
- resolved prefill kernel path
- resolved decode kernel path
- active `decodeMode` and `batchGuardReason`
- speculation / multi-token decode state
- actual loaded weight/materialization dtype for the hot ops

4. Compare boundary-by-boundary and stop at first divergence:
- embeddings
- post input norm
- Q/K/V pre-RoPE
- Q/K post-RoPE
- attention output
- FFN output
- final logits

5. Once token IDs or embeddings match, stop changing prompt wrappers or harness formatting until later evidence requires it.

6. For quantized failures, run one F16 or source-precision control before changing quantized kernels.
- F16/source-precision good + quantized bad => quantized path issue
- F16/source-precision bad + quantized bad => shared conversion/layout/runtime issue

7. Prefer one config-driven probe over one new theory.

For decode performance regressions, classify the wall before editing kernels:
- `decodeRecordMs` high => GPU compute or recording path
- `decodeSubmitWaitMs` / `decodeReadbackWaitMs` high => orchestration or readback path
- `singleTokenReadbackWaitMs` high on a fair parity lane => likely not a phase-math problem first

Reference workflow: `docs/debug-playbook.md`
Reusable report template: `docs/debug-investigation-template.md`
Canonical protocol source: `docs/agents/debug-protocol.md`

## Conversion Completion Discipline

See also: `docs/agents/conversion-protocol.md`

Do not report conversion success unless all of these are true:
- process exited successfully
- `manifest.json` exists
- shard set is present
- conversion report is valid

If shards exist without a manifest, classify the output as interrupted/incomplete and clean or overwrite it before retrying. Do not treat it as a reference artifact.

## Fast Triage

```bash
# Discover checked-in runtime profiles and trace/probe signals
npm run cli -- profiles --json

# Primary debug run (auto surface = node-first transport; browser fallback only when node transport is unavailable)
npm run debug -- --config '{"request":{"modelId":"MODEL_ID"},"run":{"surface":"auto"}}' --runtime-profile profiles/verbose-trace --json

# Verify pass/fail with inference suite
npm run verify:model -- --config '{"request":{"workload":"inference","modelId":"MODEL_ID"},"run":{"surface":"auto"}}' --runtime-profile profiles/verbose-trace --json

# Force browser relay for mobile/WebGPU parity checks
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"diagnostics/debug-logits"},"run":{"surface":"browser","browser":{"channel":"chrome","console":true}}}' --json
```

## Runtime Overrides (Config-First)

Use runtime JSON patches instead of ad-hoc flags:

```bash
npm run debug -- \
  --config '{"request":{"modelId":"MODEL_ID"},"run":{"surface":"auto"}}' \
  --runtime-config '{"shared":{"debug":{"trace":{"enabled":true,"categories":["attn","ffn"],"maxDecodeSteps":2}}},"inference":{"generation":{"maxTokens":8},"sampling":{"temperature":0,"topK":1,"topP":1,"repetitionPenalty":1,"greedyThreshold":0},"session":{"decodeLoop":{"batchSize":1,"stopCheckMode":"batch","readbackInterval":1}}}}' \
  --json
```

Notes:
- Decode cadence lives under `runtime.inference.session.decodeLoop`, not `runtime.inference.batching`.
- Token budget lives under `runtime.inference.generation.maxTokens`, not `sampling.maxTokens`.

## Focused Probes

```bash
# Broad trace-heavy debug run
npm run debug -- --config '{"request":{"modelId":"MODEL_ID"},"run":{"surface":"auto"}}' --runtime-profile profiles/verbose-trace --json

# Logit-focused browser relay run
npm run debug -- --config '{"request":{"modelId":"MODEL_ID","runtimeProfile":"diagnostics/debug-logits"},"run":{"surface":"browser","browser":{"channel":"chrome","console":true}}}' --json

# Minimal deterministic decode probe
npm run debug -- \
  --config '{"request":{"modelId":"MODEL_ID"},"run":{"surface":"auto"}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":16},"sampling":{"temperature":0,"topK":1,"topP":1,"repetitionPenalty":1,"greedyThreshold":0},"session":{"decodeLoop":{"batchSize":1,"stopCheckMode":"batch","readbackInterval":1}}}}' \
  --json
```

Notes:
- Use `debug` for trace/probe work and `verify:model` for pass/fail gates.
- For throughput sweeps or batching/readback tuning, switch to `doppler-perf`; do not overload debug runs with benchmark methodology.

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

- `src/cli/doppler-cli.js`
- `src/tooling/command-api.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-browser-command-runner.js`
- `src/inference/browser-harness.js`
- `src/config/runtime/profiles/verbose-trace.json`
- `docs/developer-guides/README.md`

## Related Skills

- `doppler-bench` for perf regression quantification
- `doppler-convert` when conversion integrity is suspected
