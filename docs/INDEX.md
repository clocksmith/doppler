# Doppler Docs Index

Primary documentation index.

## Install and run

- [Getting Started](getting-started.md) — published installation and first use.
- [Run API](api/root.md) — signed-Capsule source facade and its release scope.
- [Compatibility API](api/compat.md) — explicit published manifest-loading path.
- [Performance and sizing](performance-sizing.md) — measured limits and prerequisites.
- [Product priorities](../GOALS.md#current-priorities) — durable goals and ownership.
- [Product acceptance](../GOALS.md#product-acceptance-and-evidence) — technical requirements and retained campaign evidence.

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
- [Release contract](model-release-platform.md) — Capsule release closure, the non-promoting release command, and activation and custody boundaries.
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
- [Potential Doppler optimizations](performance/potential-optimizations.md) - single prepared-execution opportunity inventory, ordered implementation and evidence gates.
- [Kernel Performance Optimization](developer-guides/16-kernel-performance-optimization.md) - phase-led GPU optimization, negative results, and parity stopping rules.
- [Kernel Benchmark Baselines](../tests/kernels/benchmarks.md) - expected kernel perf ranges and reference baselines.
- [Benchmark Methodology](benchmark-methodology.md) - fairness and claim publication policy.
- [Vendor Registry](../benchmarks/vendors/README.md) - cross-product benchmark contracts and tooling.
- [Release Matrix](release-matrix.md) - generated model/platform support snapshot.

## Training and Distillation

- [Training Handbook](training-handbook.md) - canonical operator workflow, gates, and artifact contract.
- [Training Artifact Policy](training-artifact-policy.md)
- [Verifier-Guided and RLVR Training Contract](rlvr-training-contract.md) - method names, rollout and reward receipts, verifier separation, and promotion gates.
- [WGSL Repair and Writer Campaign](https://github.com/clocksmith/gamma/blob/main/projects/distillation/wgsl/campaign/README.md) - moved to Gamma: V8 student replay, V9-V13 repair, writer v1-v3, and source catalog v2 receipts, tools, and policies.
- [WGSL Writer v2 Result](status/wgsl-writer-v2-result-2026-07-14.md) - Qwen 3.5 9B seed 47 is selected, three-seed semantic confirmation passes at 95.83% mean, and the selected LoRA has exact Transformers-to-Doppler completion parity; external promotion and general writing remain blocked.
- [Qwen 3.5 9B Doppler-Native Training Parity Design](status/qwen35-9b-doppler-native-training-parity-design-2026-07-12.md) - SAME-R backend-parity gates, implemented F16 frozen-weight mechanics, and explicit Qwen hybrid-graph blockers.
- [Training Migrations](training-migrations.md)

## Style Guides

- [Style Guides](style/README.md)

## Specs and Source Readmes

- [Benchmark Schema](../benchmarks/benchmark-schema.json)
- [Training Engine](../src/experimental/training/GUIDE.md)
- [Inference Guide](../src/inference/GUIDE.md)
- [Kernel Tests](../tests/kernels/GUIDE.md)
