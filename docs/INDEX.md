# Doppler Docs Index

Primary documentation index.

## Install and run

- [Getting Started](getting-started.md) — published installation and first use.
- [Run API](api/root.md) — signed-Capsule source facade and its release scope.
- [Compatibility API](api/compat.md) — explicit published manifest-loading path.
- [Performance and sizing](performance-sizing.md) — measured limits and prerequisites.
- [Implementation checklist](../TODO.md) — the sole active sequence and completion state.
- [Product priorities](../GOALS.md#current-priorities) — durable goals and ownership.
- [Product contracts](goals.md) — technical requirements and retained campaign evidence.

## Supported operations and evidence

- [API index](api/index.md) — public operations and stability boundaries.
- [Generation API](api/generation.md) — lower-level text generation.
- [Choice scoring](api/choice-scoring.md) — contextual single-token labels, raw logits, separate qualification.
- [Model support matrix](model-support-matrix.md) — generated model verification scope.
- [Model support inventory](model-support-inventory.md) — evidence gaps.
- [Subsystem support](subsystem-support-matrix.md) — public, experimental, and internal-only status.
- [Model competition scoreboard](model-competition-scoreboard.md) — comparison receipts.
- [Release matrix](release-matrix.md) — model/platform evidence snapshot.

## Model preparation

- [Library engineering workflows](developer-guides/library-engineering-workflows.md) — repository conversion, verification, and benchmarks.
- [Developer guides](developer-guides/README.md) — task-oriented extension playbooks.
- [Onboarding tooling](onboarding-tooling.md) — source inspection, checks, and scaffolding.
- [RDRR format](rdrr-format.md) — runtime artifact specification.
- [Conversion/runtime contract](conversion-runtime-contract.md) — ownership and overrides.
- [ModelIR v2 and source-truth Rig](model-ir-v2-source-truth-forge.md) — semantics and provenance.
- [Model promotion](model-promotion-playbook.md) and [registry workflow](registry-workflow.md) — artifact/catalog/hosting synchronization.
- [Direct-source proof lanes](direct-source-proof-lanes.md) — additional source-format promotion requirements.

## Integration

- [Architecture](architecture.md#technical-diagrams) — component diagrams, Capsule execution, partition resource lifetimes, and host boundaries.
- [Resident partitions](distribution/resident-partition-execution.md) — public factories, implemented restrictions, and remaining physical acceptance.
- [Program Bundles](integration/program-bundle.md) — closed export and optional provider integration.
- [Product integration qualification](product-integration-qualification.md) — identity and application outcomes.
- [Provider conformance](provider-conformance.md) — separately scoped browser/Node/provider support.
- [Runtime ownership](runtime-ownership.md) — choosing an execution owner with evidence.
- [Bun qualification](bun-product-qualification.md) — experimental host evidence and promotion.
- [Loaders](api/loaders.md), [orchestration](api/orchestration.md), and [advanced root exports](api/advanced-root-exports.md) — lower-level APIs.

## Advanced operations

- [Rig/Run naming](rig-run-naming.md) — names and compatibility.
- [Component index](component-index.md) — generated authority map.
- [Pipeline contract](pipeline-contract.md) and [config](config.md) — normative execution behavior.
- [CLI reference](cli.md), [tooling API](api/tooling.md), and [operations](operations.md) — engineering commands and diagnostics.
- [Incremental Capsule streams](capsule-streaming.md) — migration and receipt transport.
- [Revocation](revocation.md) — explicit trust and release-history boundaries.
- [Optional release operations](model-release-platform.md) — retained commercial hypotheses and release contracts.
- [Optional adoption plan](executable-model-adoption-plan.md) — independent integration proof, not another active queue.
- [LoRA format](lora-format.md), [experimental tooling](api/tooling-experimental.md), [diffusion](api/diffusion.md), and [energy](api/energy.md) — separately scoped capabilities.
- [Generated exports](api/reference/exports.md) — machine-derived inventory.

## Historical and experimental campaigns

- [Archived package changelog through 0.4.15](status/archive/package-changelog-through-0.4.15.md) — retained release history.
- [Search starter](../examples/document-search/README.md) — versioned example; fresh-install metadata expired.
- [Glimmer campaign](programs/glimmer-architectural-generalization.md) — lowered candidates with failing source parity; separate capability gates.
- [MoM layer draft](distribution/mom-layer-draft.md) — experimental coordination/research proposal.

## Testing and Benchmarks

- [Testing](testing.md) - testing index.
- [Testing Runbook](testing-runbook.md) - operational test execution.
- [Kernel Testing Design](kernel-testing-design.md) - kernel correctness design guidance.
- [Kernel Performance Optimization](developer-guides/16-kernel-performance-optimization.md) - phase-led GPU optimization, negative results, and parity stopping rules.
- [Kernel Benchmark Baselines](../tests/kernels/benchmarks.md) - expected kernel perf ranges and reference baselines.
- [Benchmark Methodology](benchmark-methodology.md) - fairness and claim publication policy.
- [Vendor Registry](../benchmarks/vendors/README.md) - cross-product benchmark contracts and tooling.
- [Release Matrix](release-matrix.md) - generated model/platform support snapshot.

## Training and Distillation

- [Training Handbook](training-handbook.md) - canonical operator workflow, gates, and artifact contract.
- [Training Artifact Policy](training-artifact-policy.md)
- [Verifier-Guided and RLVR Training Contract](rlvr-training-contract.md) - method names, rollout and reward receipts, verifier separation, and promotion gates.
- [WGSL Student Replay v8 Receipt](status/wgsl-student-replay-v8-2026-07-11.md) - terminal rejected result, preserved mechanics evidence, and held-out failure boundary.
- [WGSL Repair v9 Status](status/wgsl-repair-v9-2026-07-11.md) - historical Radeon-verified corpus and optimizer-harness receipt.
- [WGSL Repair v10 Result](status/wgsl-repair-v10-2026-07-12.md) - Qwen 3.5 9B seed-11 SFT improves family-disjoint compiler-repair pass@1 from 8.36% to 88.29%, with semantic and promotion limits retained.
- [WGSL Repair v12 Controlled-Lane Design](status/wgsl-repair-v12-design-2026-07-12.md) - full seed-ordered anchor/external/random controls plus short/long repair strata; harness-ready with no V12 outcome.
- [WGSL Repair v12 Adapter Portability](status/wgsl-repair-v12-adapter-portability-2026-07-13.md) - preserved external20 artifacts plus exact prompt, completion, and aligned-logit parity for all three adapters after the split-SwiGLU correction; no seed-selection authority.
- [WGSL Repair v13 Seed Selection](status/wgsl-repair-v13-seed-selection-2026-07-14.md) - frozen semantic ranking selects external20 seed 29 at 4/6 tasks; confirmation, promotion, and WGSL Doctor remain blocked.
- [WGSL Repair v13 Blind Seed-Confirmation Freeze](status/wgsl-repair-v13-seed-confirmation-freeze-2026-07-14.md) - commit-derived eight-of-twelve semantic population protocol frozen before seed-29 confirmation inference.
- [WGSL Repair v13 Seed-Confirmation Readiness](status/wgsl-repair-v13-seed-confirmation-readiness-2026-07-14.md) - balanced commit-derived population and strict seed-29-only gate frozen before reference qualification or candidate inference.
- [WGSL Repair v13 Seed-Confirmation Result](status/wgsl-repair-v13-seed-confirmation-result-2026-07-14.md) - seed 29 passes 8/8 semantic tasks and 24/24 dispatch variants; promotion, WGSL Doctor, and full-shader writing remain separate.
- [WGSL Writer v1 Mechanics Freeze](status/wgsl-writer-v1-mechanics-freeze-2026-07-14.md) - separate complete-shader specification/interface contract and executable semantic harness frozen before reference or model execution.
- [WGSL Writer v1 Reference Mechanics](status/wgsl-writer-v1-reference-mechanics-2026-07-14.md) - reference shaders pass 3/3 tasks and 9/9 primary variants; model capability and candidate execution remain unclaimed.
- [WGSL Writer v1 Zero-Shot Diagnostic Freeze](status/wgsl-writer-v1-diagnostic-freeze-2026-07-14.md) - one matched visible-mechanics submission each for Qwen 9B base and V13 seed-29 initialization, with no selection authority.
- [WGSL Writer v1 Zero-Shot Diagnostic Result](status/wgsl-writer-v1-diagnostic-result-2026-07-14.md) - both frozen initializations score 0/3 response, compile, and semantic pass; no current writer capability or repair-to-writer transfer is established.
- [WGSL Writer v2 Result](status/wgsl-writer-v2-result-2026-07-14.md) - Qwen 3.5 9B seed 47 is selected, three-seed semantic confirmation passes at 95.83% mean, and the selected LoRA has exact Transformers-to-Doppler completion parity; external promotion and general writing remain blocked.
- [WGSL Writer V3 Plan](wgsl-writer-v3-plan.md) - sequenced plan for an executable compute/render shader package, Chromium reference qualification, semantic evaluation, matched training, exact browser-artifact promotion, and a gated product surface.
- [WGSL Writer v3 Mechanics Freeze](status/wgsl-writer-v3-mechanics-freeze-2026-07-16.md) - Freezes an executable compute/render/multi-pass package contract and family-disjoint general-authoring plan; browser execution, reference oracles, corpus materialization, training, and every capability claim remain blocked.
- [WGSL Writer v3 Reference Qualification](status/wgsl-writer-v3-reference-qualification-2026-07-18.md) - Qualifies the four frozen compute/render/multi-pass reference packages with deterministic replay, semantic oracles, cleanup evidence, and an identity-bound AMD/Vulkan Chromium receipt; corpus, training, and capability claims remain blocked.
- [WGSL Writer v3 Campaign Reconciliation](status/wgsl-writer-v3-campaign-reconciliation-2026-07-19.md) - Preserves the original frozen gate, classifies later policies as development evidence, and requires a prospective family-disjoint materialization transition.
- [WGSL Repair v13 Semantic Readiness (pre-selection)](status/wgsl-repair-v13-semantic-readiness-2026-07-14.md) - passing adapter portability and reference dispatch mechanics admit frozen calibration/checkpoint selection.
- [WGSL Repair v13 Semantic Contract (historical)](status/wgsl-repair-v13-semantic-design-2026-07-13.md) - original frozen requirements and immutable V1 blocked receipt before portability recovery.
- [Qwen 3.5 9B Doppler-Native Training Parity Design](status/qwen35-9b-doppler-native-training-parity-design-2026-07-12.md) - SAME-R backend-parity gates, implemented F16 frozen-weight mechanics, and explicit Qwen hybrid-graph blockers.
- [WGSL ML Kernel Source Catalog v2](status/wgsl-kernel-source-catalog-v2-2026-07-12.md) - Pinned training, reference-only, and quarantined WebGPU ML sources, including MLC WebLLM.
- [Training Migrations](training-migrations.md)

## Style Guides

- [Style Guides](style/README.md)

## Specs and Source Readmes

- [Benchmark Schema](../benchmarks/benchmark-schema.json)
- [Training Engine](../src/experimental/training/GUIDE.md)
- [Inference Guide](../src/inference/GUIDE.md)
- [Kernel Tests](../tests/kernels/GUIDE.md)
