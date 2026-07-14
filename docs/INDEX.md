# Doppler Docs Index

Primary documentation index.

## Start Here

- [Getting Started](getting-started.md) - canonical first-run workflow.
- [Goals](goals.md) - compact product and technical contract for Doppler's mainline work.
- [Developer Guides](developer-guides/README.md) - task-oriented extension playbooks for adding models, kernels, commands, and pipeline features.
- [Performance and Sizing](performance-sizing.md) - hardware tiers and planning guidance.

## Core Runtime Docs

- [Architecture](architecture.md) - system model and boundaries.
- [Pipeline Contract](pipeline-contract.md) - command-to-output runtime contract boundaries.
- [Config](config.md) - kernel paths, config behavior, and runtime contract notes.
- [CLI Reference](cli.md) - command flags, config inputs, surface selection, and examples.
- [RDRR Format](rdrr-format.md) - runtime artifact spec.
- [Direct-source proof lanes](direct-source-proof-lanes.md) - phased promotion
  plan for `safetensors` and `gguf` as additional canonical proof lanes.
- [LoRA Format](lora-format.md) - adapter manifest spec.
- [Conversion Runtime Contract](conversion-runtime-contract.md) - conversion-static vs runtime-overridable ownership.
- [Model Promotion Playbook](model-promotion-playbook.md) - canonical sync workflow for repo metadata, external-volume RDRR artifacts, and Hugging Face hosting.
- [Model Roadmap](model-roadmap.md) - editorial model priorities and status, separate from implementation lanes.
- [Model Support Matrix](model-support-matrix.md) - generated model status table from catalog, conversion coverage, and quickstart metadata.
- [Model Competition Scoreboard](model-competition-scoreboard.md) - generated model/platform/Transformers.js evidence ledger from catalog support and benchmark receipts.
- [Subsystem Support Matrix](subsystem-support-matrix.md) - generated support-tier contract for public, experimental, and internal-only subsystem surfaces.
- [Registry Workflow](registry-workflow.md) - hosted catalog validation and Hugging Face publication workflow.
- [Operations](operations.md) - troubleshooting and debug workflows.
- [Doppler Program Bundle](integration/program-bundle.md) - closed portable
  model-program export shape for browser, Node provider, Doe.js capture, and
  Doe backend lowering.
- [MoM Layer Draft](distribution/mom-layer-draft.md) - implementation-side draft
  for Cross Family Router, debate/adjudication, recursive MoM receipts,
  residency metadata, and Debate-adjudicated Distillation exports.

## Public API Docs

- [API Docs Index](api/index.md) - canonical public API navigation.
- [Root API](api/root.md) - app-facing `doppler` facade.
- [Advanced Root Exports](api/advanced-root-exports.md) - root-level loaders, adapters, and advanced exports.
- [Loaders API](api/loaders.md) - explicit loader and manifest/bootstrap helpers.
- [Orchestration API](api/orchestration.md) - KV cache, routers, adapters, and logit merge helpers.
- [Generation API](api/generation.md) - lower-level text pipeline surface.
- [Diffusion API](api/diffusion.md) - experimental diffusion/image pipeline surface.
- [Energy API](api/energy.md) - experimental energy pipeline surface.
- [Tooling API](api/tooling.md) - command-runner and tooling surface.
- [Experimental Tooling API](api/tooling-experimental.md) - experimental browser-conversion and P2P helper surface.
- [Generated Export Inventory](api/reference/exports.md) - machine-derived export inventory from package entrypoints.

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
- [WGSL Repair v12 Adapter Portability](status/wgsl-repair-v12-adapter-portability-2026-07-13.md) - preserved external20 artifacts, repaired prompt and LoRA import mechanics, and the retained failed trainer-to-Doppler behavioral parity gate.
- [WGSL Repair v13 Semantic Contract](status/wgsl-repair-v13-semantic-design-2026-07-13.md) - frozen dispatch, CPU-oracle, bounds, metamorphic, and regression requirements; semantic evaluation and WGSL Doctor remain blocked.
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
