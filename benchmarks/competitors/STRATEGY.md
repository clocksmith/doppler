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
