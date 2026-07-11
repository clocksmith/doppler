# Doppler Performance and Correctness Roadmap (Execution-Accurate, NEXT-only)

Status note: this file has been corrected to reference only modules, functions, and rules that currently exist in the repository. It is documentation-only and does not change runtime behavior.

## Active Metal Decode Work

- Close the Qwen 3.5 2B Metal comparison with at least 20 interleaved 512-token
  or longer Doppler/Transformers.js pairs and a paired 95% confidence interval.
  If the interval crosses zero and the median difference is below 0.5%, record
  parity within measurement noise and stop projection tuning.
- For a statistically supported residual, compare encoded dispatches per token
  before adding another math kernel. Pursue decoder-operation fusion only when
  the reference path proves a lower launch count.
- Profile the untouched attention share: RoPE, softmax, KV-cache reads, and
  attention output assembly. Inspect cache layout and access contiguity when its
  decode share exceeds the reference engine.

The Q4_K decomposition, dtype, workgroup geometry, shared-memory, submission
cadence, and fused LM-head experiments are durable findings in
`docs/developer-guides/16-kernel-performance-optimization.md`, not open work in
this file.

## 200-Word Execution Goal

Build and execute an implementation pass that removes speculative roadmap entries and replaces them with verifiable work on existing codepaths only, then makes the minimum safe edits required for parity and traceability. This work must stay grounded in:
`src/inference/pipelines/text.js`, `src/inference/pipelines/text/model-load.js`, `src/inference/pipelines/text/execution-plan.js`, `src/inference/pipelines/text/generator-steps.js`, `src/gpu/kernels/sample.js`, `src/inference/decode-ring.js`, `src/memory/buffer-pool.js`, `src/gpu/partitioned-buffer-pool.js`, `src/loader/doppler-loader.js`, `src/loader/shard-cache.js`, and `src/loader/tensors/tensor-loader.js`.

Within this pass, for each phase:
1) map every change to concrete function-level anchors,  
2) define explicit acceptance criteria and failure fallbacks,  
3) preserve contract invariants (explicit-over-implicit configuration, manifest-first, config-only policy ownership, no runtime JavaScript defaults, and no hidden fallback behavior),  
4) keep browser/Node behavior aligned, and  
5) attach existing tests as gates.

Required gates: `tests/inference/execution-plan.test.js`, `tests/inference/sampling-topk-fast-path.test.js`, `tests/inference/ffn-mixed-q4k-materialization-contract.test.js`, `tests/inference/decode-ring.test.js`, and `tests/gpu/sample-cleanup.test.js`.

Completion condition: no missing-module references remain, every phase item is anchored to real callsites, and all listed gates are ready to run with deterministic expected outcomes before implementation proceeds beyond docs.

## Scope and Contract

Target optimization and hardening tracks should stay on existing call-sites:

- Inference load + execution graph: `src/inference/pipelines/text.js`, `src/inference/pipelines/text/model-load.js`, `src/inference/pipelines/text/execution-plan.js`
- Sampling dispatch + readback: `src/inference/pipelines/text/generator-steps.js`, `src/gpu/kernels/sample.js`
- Q4_K materialization and loader hooks: `src/loader/doppler-loader.js`, `src/loader/shard-cache.js`, `src/loader/tensors/tensor-loader.js`
- Buffer lifecycle and ring state: `src/memory/buffer-pool.js`, `src/gpu/partitioned-buffer-pool.js`, `src/inference/decode-ring.js`

Primary invariants in this plan:

- Preserve existing contracts from `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, and `docs/style/config-style-guide.md`.
- No behavior changes through speculative modules or new surfaces.
- Keep command parity (browser + Node + CLI) by editing existing runtime contracts only.
- Any regression path must be explicit and test-addressed in current suites.

## What is already in place (so the plan avoids re-implementing)

- `src/inference/pipelines/text.js` already calls into the load/build sequence:
  - `applyExecutionV1RuntimeConfig(...)`
  - `parseModelConfig(...)`
  - `applyModelBatchingRuntimeDefaults(...)`
  - `resolveKernelPathState(...)`
  - `_resolveLayerPipeline(...)`
  - `compileExecutionPlanState(...)`
- `src/inference/pipelines/text/model-load.js` includes:
  - `resolveKernelPathState(...)`
  - `activateKernelPathState(...)`
  - kernel-path policy + resolve helpers (`resolveKernelPathPolicy` / `resolveKernelPath`) and execution-plan integration
- `src/inference/pipelines/text/execution-plan.js` includes execution-plan compile/activation/fallback plumbing.
- `src/gpu/kernels/sample.js` already includes:
  - `isGPUSamplingAvailable(...)`
  - `resolveSampleVariants(...)`
  - `runArgmax(...)`, `runGPUSample(...)`
  - `recordArgmax(...)`, `recordGPUSample(...)`
- `src/inference/pipelines/text/generator-steps.js` already consumes fused/host sampling hooks:
  - `shouldUseFusedDecodeSampling(...)`
  - `recordArgmax(...)`
  - `recordGPUSample(...)`
  - `recordLogitsGPU(...)`
  - `readSampledTokenFromStagingBuffer(...)`
- Q4_K pipeline support is already present in current loader code:
  - `src/loader/doppler-loader.js`: `setQ4KConfig`, `setCustomShardLoader`, `#shouldStreamUploadToGPU`, `#assembleShardDataToGpuBuffer`, `#assembleShardData`
  - `src/loader/tensors/tensor-loader.js`: `shouldUseFusedQ4K`, `resolveQ4KLimitFallback`, `loadQ4KFused`, `loadQ4KDequant`, `loadQ4KMixed`
  - `src/loader/shard-cache.js`: `setCustomLoader`, `loadRange`, `streamRange`

## Non-existent targets to remove from execution plan

- `src/client/runtime/gpu-arena.js`
- `src/client/runtime/pool-allocator.js`
- `src/worker/opfs-loader-worker.js`
- New WGSL file sets for hypothetical fused kernels not yet wired in repo
- New module folders under `src/client/runtime` suggested by prior draft

---

## Phase 0 — Baseline Contract Lock (Do this before any edits)

Goal: freeze the exact behavior contract and make deltas measurable.

Exact touchpoints:

- `docs/style/general-style-guide.md` / `docs/style/javascript-style-guide.md` / `docs/style/config-style-guide.md`
- `src/inference/pipelines/text.js` (`loadModel` call flow)
- `src/inference/pipelines/text/model-load.js` (`resolveKernelPathState`, `compileExecutionPlanState`, `activateKernelPathState`)
- `src/inference/pipelines/text/execution-plan.js` (plan compile + fallback activation)

Acceptance checks:

1. Confirm all milestones in this file map to symbols in the existing tree.
2. Confirm there is no plan text that introduces new public module surfaces.
3. Confirm command/runtime parity text references only existing `runtimeConfig`/`runtimeProfile`/`configChain` fields.

Gate for phase completion:

- `NEXT.md` sections below must only reference concrete files + functions above and no speculative modules.

---

## Phase 1 — Execution-path verification and readback contract tightening

Goal: document and validate that existing fused/host sampling and execution-plan gating already match intended runtime contracts.

Exact touchpoints:

- `src/inference/pipelines/text/generator-steps.js`
  - `shouldUseFusedDecodeSampling`
  - `recordArgmax`
  - `recordGPUSample`
  - `recordLogitsGPU`
  - `readSampledTokenFromStagingBuffer`
- `src/gpu/kernels/sample.js`
  - `isGPUSamplingAvailable`
  - `resolveSampleVariants`
  - `runArgmax`, `runGPUSample`
  - `recordArgmax`, `recordGPUSample`
- `src/inference/decode-ring.js`
  - `ensure`, `acquire`, `advance`, `release`
- `src/memory/buffer-pool.js`
  - buffer acquire/release lifecycle

Plan actions:

- Audit callsites so every sampling execution path is explicitly tied to:
  - GPU sample variant resolution
  - staged token readback path
  - explicit fallback path into host sampling when GPU sampling variant is unavailable
- Add/confirm contract comments only in existing surfaces where behavior toggles are decided (no new modules).
- Ensure ring lifecycle is called from decode loop in the same semantics currently observed by generation and no hidden staging path bypasses it.

Acceptance criteria:

- Fixed prompt + fixed seed produces identical behavior for existing fallback default.
- GPU sampling path and host sampling path are both documented with explicit gates and failure fallback in this doc and in tests.
- Readback staging does not mutate sequence state when sampling disabled.

Existing regression anchors:

- `tests/inference/sampling-topk-fast-path.test.js`
- `tests/gpu/sample-cleanup.test.js`
- `tests/inference/generator-token-id-hint-contract.test.js`

---

## Phase 2 — Kernel-path and execution-plan determinism

Goal: make kernel-path activation behavior auditable and stable across runtime sessions.

Exact touchpoints:

- `src/inference/pipelines/text/model-load.js`
  - `resolveKernelPathPolicy`
  - `resolveKernelPathState`
  - `activateKernelPathState`
  - `compileExecutionPlanState`
  - `resolveKernelPath`
- `src/inference/pipelines/text/execution-plan.js`
  - execution-plan compilation + active plan activation
  - fallback activation helpers
- `src/inference/pipelines/text.js`
  - `loadModel` sequence order around runtime compile
- `src/rules/` entries that feed kernel-path + sampling behavior
  - especially `src/rules/*/*.rules.json` used by loader and kernels

Plan actions:

- Capture the exact state boundaries at:
  - kernel-path resolution
  - execution-plan selection
  - fallback execution-plan activation
- Validate that selection remains based on config + rules, not implicit branch defaults.
- Keep fallback semantics explicit (no silent remap) and visible in logs/diagnostics where existing debug hooks already exist.

Acceptance criteria:

- For one supported model workload, kernel-path source and active plan are stable over identical config and manifest inputs.
- Existing compile-fallback behavior still works and remains recoverable to host-safe path when needed.
- No runtime family detection in JS for kernel selection.

Existing regression anchors:

- `tests/inference/execution-plan.test.js`
- `tests/inference/model-load-batching-defaults.test.js`

---

## Phase 3 — Q4_K materialization, fused-vs-dequant contracts, and loader boundaries

Goal: close any gaps between documented and actually applied Q4_K policy using current loader hooks.

Exact touchpoints:

- `src/loader/doppler-loader.js`
  - `setQ4KConfig`
  - `setCustomShardLoader`
  - `#shouldStreamUploadToGPU`
  - `#assembleShardDataToGpuBuffer`
- `src/loader/shard-cache.js`
  - `setCustomLoader`
  - `loadRange`
  - `streamRange`
- `src/loader/tensors/tensor-loader.js`
  - `shouldUseFusedQ4K`
  - `resolveQ4KLimitFallback`
  - `loadQ4KFused`
  - `loadQ4KDequant`
  - `loadQ4KMixed`

Plan actions:

- Verify the exact dispatch condition used by each of:
  - fused loader
  - dequant loader
  - mixed loader
  and trace it back to `q4k` runtime fields in resolved model/session config.
- Ensure `setQ4KConfig` + custom shard loader wiring is only used in load scenarios already covered by existing callsites.
- Validate stream-upload gating conditions around `#shouldStreamUploadToGPU` remain deterministic under OPFS/stream constraints.

Acceptance criteria:

- Q4_K mode choice for each tensor load is explicit and testable from resolved config + runtime flags.
- Fallback from fused/dequant/mixed is explicit and reversible.
- No fallback path is silently masked (must be observable in existing diagnostics/tests).

Existing regression anchors:

- `tests/inference/ffn-mixed-q4k-materialization-contract.test.js`
- `tests/loader/layer-loader-conv-precision.test.js`
- `src/loader/doppler-loader.js` integration checks already exercised by current loader smoke flow

---

## Phase 4 — Loading and streaming observability hardening

Goal: document stream-loading boundaries and ensure staged data flow is bounded and deterministic using existing APIs.

Exact touchpoints:

- `src/loader/shard-cache.js` (`loadRange`, `streamRange`, `setCustomLoader`)
- `src/loader/doppler-loader.js` (`#assembleShardData`, `#assembleShardDataToGpuBuffer`)
- `src/inference/pipelines/text/model-load.js` model-session loader application points (`setCustomShardLoader` wiring)

Plan actions:

- Document exact sequence of stream range load -> custom shard loader -> tensor upload -> execution compile.
- Capture guard conditions for when loader uses stream upload vs non-stream path.
- Add a clear rollback note in this doc for any change that impacts memory pressure or OPFS behavior.

Acceptance criteria:

- No load path change occurs without an explicit condition anchored to existing loader fields and functions.
- Existing tests that exercise streaming and loading order remain stable.

Existing regression anchors:

- `tests/loader/doppler-loader` family and existing integration tests in loader suite
- Any loader-related GPU memory regression should be caught by `tests/gpu/sample-cleanup.test.js` + decode flow tests

---

## Phase 5 — Buffer lifecycle and decode-ring audit

Goal: align buffer lifecycle and decode ring usage with failure-safe cleanup invariants.

Exact touchpoints:

- `src/memory/buffer-pool.js` (`acquireBuffer`, `releaseBuffer`, `destroy`)
- `src/gpu/partitioned-buffer-pool.js` (partitioned acquisition/release boundaries)
- `src/inference/pipelines/text/generator-steps.js` (sampling and readback buffer ownership around sample staging)
- `src/inference/decode-ring.js` (`ensure`, `acquire`, `advance`, `release`)
- `src/client/runtime/source-runtime.js` and `src/client/runtime/model-session.js` where runtime session state is assembled and torn down

Plan actions:

- Add explicit ownership notes in this plan for each buffer allocation/readback boundary currently used in sample/readback and model load.
- Verify pool release and token-ring transitions are paired for both prefill and decode branches.
- Ensure decode-ring disable/enable semantics remain unchanged and covered by existing decode-loop behavior.

Acceptance criteria:

- No new resource-lifetime assumptions introduced without mapping to an existing owner.
- Existing tests continue to assert cleanup behavior after early exit/throw paths.
- Regression boundary for pool churn and ring allocation remains non-regressive in baseline.

Existing regression anchors:

- `tests/gpu/sample-cleanup.test.js`
- `tests/inference/decode-ring.test.js`

---

## Implementation discipline while editing code later

- Use `src/config/runtime/profiles/*` and schema-backed fields for any new runtime policy.
- Do not add new command inputs for behavior-changing tuning.
- Keep host fallback explicit in docs and tests whenever GPU variants are gated by capability.
- Any behavior change touching:
  - kernel-path state
  - sampling mode
  - Q4_K materialization
  - streaming tensor upload policy
  must include one existing test file and one deterministic reproducibility artifact path in CI or harness flow.

## Exit criteria (plan complete only when all gates pass)

1. Every milestone in this file maps to an existing function callsite.
2. No references remain to non-existent modules/files.
3. Existing test suites for execution-plan, sampling, Q4_K materialization, and decode-ring have passable gates tied to concrete edits.
4. Next execution wave can start with a single-file edit pass scoped by these phases.
