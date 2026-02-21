# Doppler Competitive Strategy Playbook

This document captures how Doppler should compete against browser AI incumbents.
It is execution-focused and tied to measurable benchmark evidence.

Scope: browser inference products listed in `registry.json`.

## Strategic Objective

Win the category of:

- Production browser inference for fixed-model applications
- Real-device performance visibility and reproducibility
- Local-first UX (fast warm starts, predictable memory, auditable behavior)

## Non-Goals

Do not optimize primary roadmap for:

- Largest model zoo breadth
- Full multimodal task coverage parity with general-purpose stacks
- Legacy no-WebGPU compatibility as the main value proposition

## Core Wedge

Doppler should compete as a combined runtime + profiling + benchmark platform.

Positioning:

- Other stacks help run models.
- Doppler helps ship, measure, and continuously improve browser inference.

## Attack Principles

1. Evidence over narrative
- Every comparative claim must map to a normalized JSON artifact in `results/`.
- If there is no artifact, the claim does not ship.

2. Real-device truth
- Benchmark on actual browser + WebGPU surfaces, not proxy backends.
- Separate cold load vs warm load in every published report.

3. Reliability first
- Prioritize mobile/browser compatibility profiles and fallback kernel paths.
- Treat GPU pipeline failures as product bugs, not user environment noise.

4. CI-enforced performance
- Use ratchet rules for regressions on TTFT, decode throughput, and memory peaks.
- Require explicit approval to change baselines upward.

5. Integration simplicity
- Keep onboarding to static assets + one command + one config.
- Avoid forcing users into extra toolchains for normal deployment paths.

## Workstream Tracking Contract (UTC)

Use this format for every strategy item update so status is auditable and comparable across authors.

Status enum:

- `not_started`: no merged or staged implementation exists
- `in_progress`: scoped work exists but acceptance criteria are incomplete
- `implemented`: code/config path exists and resolves in runtime
- `validated_local`: reproduced with pinned local command/config/model
- `validated_ci`: covered in CI with explicit pass/fail gates
- `released`: merged and included in release notes/default docs
- `blocked`: waiting on dependency/decision/external capability

Required fields on each update entry:

- `status`
- `owner`
- `updatedUtc` (ISO8601 UTC, e.g. `2026-02-21T01:51:20Z`)
- `evidence` (file + line, and/or command summary)
- `etaUtc` (or `TBD`)
- `blocker` (or `none`)

Update rules:

- Use UTC timestamps only; do not require commit hashes in this file.
- If status changes, append a new timestamped line rather than rewriting history.
- Evidence must be concrete (`path:line` or command output summary), not prose-only.

## Technical Architecture Attack Plan

To definitively crush incumbent engines in raw performance and stability, Doppler must exploit specific WebGPU and JS boundary advantages. These high-ROI engine improvements are tracked here and should be migrated to concrete GitHub issues or execution branches.

### 1. Range-Aware Selective Widening (Mixed Precision Policy)
**Objective:** Prevent mathematical collapse without paying a global bandwidth penalty.
- **Problem:** Strict F16 computation collapses on un-clipped models (e.g. Gemma 3) where vectors naturally overflow `65,504`. Strict F32 halves the memory bandwidth and destroys generation speed.
- **Tactic:** Implement a `selective_f32_windows` kernel preset. Keep bounded operations (GEMV dot-products) packed tightly in F16 to maximize bandwidth. Strategically widen vulnerable reduction operations (Residual Adds, RMSNorm, Softmax) to F32 purely for the calculation window.
- **Status:** [~] In Progress
- **Tracking:**
  - `status`: `implemented` (Gemma 3 mitigation path), `in_progress` (selective widening policy)
  - `owner`: `runtime-kernels`
  - `updatedUtc`: `2026-02-21T01:51:20Z`
  - `evidence`: `src/config/presets/models/gemma3.json:42`, `src/config/presets/kernel-paths/gemma3-f16-f32a.json:2`, `models/curated/gemma-3-270m-it-f16-f32a/manifest.json:73`
  - `etaUtc`: `TBD`
  - `blocker`: `none`

### 2. Deferred Rounding Windows for State Updates
**Objective:** Minimize quantization noise drift during deep decoding.
- **Problem:** Constantly converting F32 intermediate activations back down to BF16 or F16 at the end of every dispatch step introduces catastrophic rounding error that drifts over 20+ layers.
- **Tactic:** Leave intermediate activations in F32 registers or fast shared memory for `N` sequential operations. Only pay the BF16/F16 quantization penalty at explicit block boundaries.
- **Status:** [ ] Not Started

### 3. Max-Subtracted Softmax with F32 Accumulation
**Objective:** Bullet-proof the attention mechanism against massive logit spikes.
- **Problem:** Naive 16-bit Softmax instantly overflows on large activations ($e^{11} > 65504$), producing a vector of `NaN` or `Infinity` that poisons the KV cache forever.
- **Tactic:** Shift all logits by subtracting the maximum value before exponentiation, and accumulate the denominator strictly in F32. This guarantees the probability distribution remains stable and normalized regardless of the input scale.
- **Status:** [x] Proved in Spike | [ ] Implementation Pending
- **Tracking:**
  - `status`: `validated_local` (spike only)
  - `owner`: `runtime-kernels`
  - `updatedUtc`: `2026-02-21T01:51:20Z`
  - `evidence`: `docs/bf16-runtime-spike.md:45`, `tools/bf16-math-spike.mjs:589`
  - `etaUtc`: `TBD`
  - `blocker`: `kernel integration + CI coverage pending`

### 4. Basis-Decomposed Paged Attention
**Objective:** Defeat linear KV Cache bandwidth scaling on immense context windows.
- **Problem:** A continuous KV cache severely limits batching and suffers from fragmentation, while linearly scanning the full VRAM history starves the GPU arithmetic logic units (ALUs).
- **Tactic:** Organize KV blocks into a "Paged" structure indexed by WebGPU. Further decompose the attention matrix mathematically to project Keys/Values into a lower-dimensional basis, trading abundant GPU ALU compute to bypass memory bandwidth bottlenecks entirely for long contexts.
- **Status:** [ ] Not Started

### 5. Always-On Runtime Finiteness Guard
**Objective:** Graceful degradation on poisoned token occurrences.
- **Problem:** A single `NaN` generated in Layer 8 gets injected directly into the KV Cache, permanently corrupting the sequence. Current guards only happen after the sequence finishes (at the logits stage).
- **Tactic:** Instrument an ultra-cheap, early-stop bitwise guard that detects non-finite values in the F16 buffers *before* they are written to the KV cache structure. If triggered, the engine drops the poisoned state and dynamically falls back to a slower, high-precision F32 path just to safely clear the "danger token" before resuming full speed.
- **Status:** [~] In Progress
- **Tracking:**
  - `status`: `in_progress`
  - `owner`: `runtime-kernels`
  - `updatedUtc`: `2026-02-21T02:55:25Z`
  - `evidence`: `src/gpu/kernels/check-finiteness.js:80`, `src/gpu/kernels/check_finiteness.wgsl:17`, `src/inference/pipelines/text/layer.js:315`, `src/inference/pipelines/text/generator.js:191`, `src/inference/pipelines/text/generator-steps.js:408`
  - `etaUtc`: `TBD`
  - `blocker`: `overflow-triggered guard regression (first-hit metadata + deterministic retry path) is not yet CI-gated`

## Gemma 3 Correctness Snapshot (UTC)

Snapshot timestamp: `2026-02-21T03:02:44Z`

1. Gemma 3 default f16-weight routing now prefers f32 activations.
- `status`: `implemented`
- `owner`: `runtime-kernels`
- `updatedUtc`: `2026-02-21T01:51:20Z`
- `evidence`: `src/config/presets/models/gemma3.json:42`
- `etaUtc`: `complete`
- `blocker`: `none`

2. Fused online f32a path for Gemma 3 is registered and selectable.
- `status`: `implemented`
- `owner`: `runtime-kernels`
- `updatedUtc`: `2026-02-21T01:51:20Z`
- `evidence`: `src/config/presets/kernel-paths/gemma3-f16-fused-f32a-online.json:2`, `src/config/presets/kernel-paths/registry.json:82`
- `etaUtc`: `complete`
- `blocker`: `none`

3. Curated f32a model artifacts exist for Gemma 3 270m and 1b.
- `status`: `implemented`
- `owner`: `conversion-runtime`
- `updatedUtc`: `2026-02-21T03:01:41Z`
- `evidence`: `models/curated/gemma-3-270m-it-f16-f32a/manifest.json:3`, `models/curated/gemma-3-270m-it-f16-f32a/manifest.json:73`, `models/curated/gemma-3-1b-it-f16-f32a/manifest.json:3`, `models/curated/gemma-3-1b-it-f16-f32a/manifest.json:73`
- `etaUtc`: `complete`
- `blocker`: `none`

4. wf16 curated manifest still points at f16a path (known risk if used as default).
- `status`: `in_progress`
- `owner`: `runtime-kernels`
- `updatedUtc`: `2026-02-21T01:51:20Z`
- `evidence`: `models/curated/gemma-3-270m-it-wf16/manifest.json:73`
- `etaUtc`: `TBD`
- `blocker`: `default migration decision not finalized`

5. NaN regression test scaffold exists but is not yet CI-gated.
- `status`: `in_progress`
- `owner`: `test-infra`
- `updatedUtc`: `2026-02-21T01:51:20Z`
- `evidence`: `tests/inference/gemma3-nan-regression.test.js:1`
- `etaUtc`: `TBD`
- `blocker`: `test is untracked and not wired into CI`

6. Always-on finiteness guard rollout has landed in code but is still being hardened.
- `status`: `in_progress`
- `owner`: `runtime-kernels`
- `updatedUtc`: `2026-02-21T02:55:25Z`
- `evidence`: `src/gpu/kernels/check-finiteness.js:107`, `src/inference/pipelines/text/generator-steps.js:408`, `src/inference/pipelines/text/generator.js:191`, `src/inference/pipelines/text/layer.js:315`
- `etaUtc`: `TBD`
- `blocker`: `needs dedicated overflow-trigger test + CI gate for deterministic fallback behavior`

7. Gemma 3 1b f32a artifact has been regenerated from HF snapshot and passes manifest integrity checks.
- `status`: `validated_local`
- `owner`: `conversion-runtime`
- `updatedUtc`: `2026-02-21T03:01:41Z`
- `evidence`: `~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/model.safetensors (dtypeCounts BF16=340)`, `models/curated/gemma-3-1b-it-f16-f32a/manifest.json:3`, `models/curated/gemma-3-1b-it-f16-f32a/manifest.json:73`, `sampled blake3 shard verification (0/14/29) matched manifest hashes`, `/tmp/bench_g3_1b_id_resolution.json (requestModelId=resultModelId=gemma-3-1b-it-f16-f32a)`
- `etaUtc`: `complete`
- `blocker`: `none`

8. Deterministic 1b kernel sweep completed locally for f16a vs f32a vs fused-f32a.
- `status`: `validated_local`
- `owner`: `runtime-kernels`
- `updatedUtc`: `2026-02-21T03:02:44Z`
- `evidence`: `/tmp/bench_g3_1b_id_gemma3_f16_f16a.json (decodeTok/s 9.22)`, `/tmp/bench_g3_1b_id_gemma3_f16_f32a.json (decodeTok/s 20.53)`, `/tmp/bench_g3_1b_id_gemma3_f16_fused_f32a_online.json (decodeTok/s 19.38)`
- `etaUtc`: `complete`
- `blocker`: `CI perf budget gates for this workload are not wired yet`

## Incumbent Attack Map

### Transformers.js

Their strength:

- Breadth of models/tasks and ecosystem gravity

Doppler attack:

- Outperform on production observability: kernel path trace, memory phase stats, deterministic bench manifests
- Outperform on repeat-visit UX: OPFS-backed warm starts
- Outperform on CI readiness: first-class benchmark contract and machine-readable outputs

### WebLLM / MLC

Their strength:

- Strong compiled performance and mature benchmark narrative

Doppler attack:

- Faster iteration on runtime behavior (kernel-path and config-level changes)
- Better developer diagnostics in the same runtime path used for shipping
- Evidence-backed mobile compatibility workarounds without recompilation loops

### MediaPipe LLM Inference

Their strength:

- Mobile-first Google ecosystem alignment

Doppler attack:

- Transparent runtime behavior (traceability and artifacted metrics)
- Wider tuning control for model/runtime combinations
- Clear reproducibility story for independent teams outside Google stack defaults

### Wllama

Their strength:

- CPU reach where WebGPU is unavailable

Doppler attack:

- Win high-value WebGPU segment on latency and throughput
- Publish clear capability matrix so unsupported devices fail fast and predictably

### Ratchet / Candle / Burn

Their strength:

- Strong systems-performance narrative in Rust ecosystems

Doppler attack:

- Lower integration friction for web product teams
- Faster browser-native debugging loop from the same command surface
- Public reproducibility registry that proves deltas on real browser targets

## 30 / 60 / 90 Day Execution

### Day 0-30

- Complete harness coverage and ingestion paths for every product in `registry.json`
- Publish first benchmark board from normalized `results/` artifacts
- Add compatibility matrix for browser/device/GPU with known failure signatures

Exit criteria:

- Every listed competitor has at least one valid normalized result
- Board generation is scriptable and repeatable

### Day 31-60

- Add CI regression gates for Doppler core metrics
- Add competitor delta snapshots (same workload, same device class)
- Add mobile kernel fallback policy and documented quirk rules

Exit criteria:

- Regressions fail CI automatically
- Mobile fallback behavior is deterministic for known failure classes

### Day 61-90

- Ship weekly benchmark publishing cadence
- Add device capability reports (feature support, viable kernel paths)
- Publish 2-3 reference deployment case studies with before/after metrics

Exit criteria:

- Weekly benchmark updates without manual cleanup
- At least one repeatable "why Doppler wins here" report per target segment

## KPI Set

Track these in benchmark summaries and release notes:

- `warm_model_load_ms` (p50/p95)
- `ttft_ms` (p50/p95)
- `decode_tokens_per_sec` (p50/p95)
- `peak_memory_mb`
- benchmark reproducibility pass rate
- mobile compatibility success rate per device class

## Governance

- Strategy messaging should align with wrapper-level narrative in:
  - `/home/x/deco/ouroboros/docs/pitch.md`
  - `/home/x/deco/ouroboros/docs/value.md`
- Doppler repo remains the implementation and measurement source of truth.
