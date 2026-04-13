# Doppler Docs Index

Primary documentation index.

## Start Here

- [Getting Started](getting-started.md) - canonical first-run workflow.
- [Developer Guides](developer-guides/README.md) - task-oriented extension playbooks for adding models, kernels, commands, and pipeline features.
- [Performance and Sizing](performance-sizing.md) - hardware tiers and planning guidance.

## Core Runtime Docs

- [Architecture](architecture.md) - system model and boundaries.
- [Pipeline Contract](pipeline-contract.md) - command-to-output runtime contract boundaries.
- [Config](config.md) - kernel paths, config behavior, and runtime contract notes.
- [CLI Reference](cli.md) - command flags, config inputs, surface selection, and examples.
- [RDRR Format](rdrr-format.md) - runtime artifact spec.
- [LoRA Format](lora-format.md) - adapter manifest spec.
- [Conversion Runtime Contract](conversion-runtime-contract.md) - conversion-static vs runtime-overridable ownership.
- [Model Promotion Playbook](model-promotion-playbook.md) - canonical sync workflow for repo metadata, external-volume RDRR artifacts, and Hugging Face hosting.
- [Model Support Matrix](model-support-matrix.md) - generated model status table from catalog, conversion coverage, and quickstart metadata.
- [Subsystem Support Matrix](subsystem-support-matrix.md) - generated support-tier contract for public, experimental, and internal-only subsystem surfaces.
- [Registry Workflow](registry-workflow.md) - hosted catalog validation and Hugging Face publication workflow.
- [Operations](operations.md) - troubleshooting and debug workflows.

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
- [Generated Export Inventory](api/reference/exports.md) - machine-derived export inventory from package entrypoints.

## Testing and Benchmarks

- [Testing](testing.md) - testing index.
- [Testing Runbook](testing-runbook.md) - operational test execution.
- [Kernel Testing Design](kernel-testing-design.md) - kernel correctness design guidance.
- [Kernel Benchmark Baselines](../tests/kernels/benchmarks.md) - expected kernel perf ranges and reference baselines.
- [Benchmark Methodology](benchmark-methodology.md) - fairness and claim publication policy.
- [Vendor Registry](../benchmarks/vendors/README.md) - cross-product benchmark contracts and tooling.
- [Release Matrix](release-matrix.md) - generated model/platform support snapshot.

## Training and Distillation

- [Training Handbook](training-handbook.md) - canonical operator workflow, gates, and artifact contract.
- [Training Artifact Policy](training-artifact-policy.md)
- [Training Migrations](training-migrations.md)

## Style Guides

- [Style Guides](style/README.md)

## Specs and Source Readmes

- [Benchmark Schema](../benchmarks/benchmark-schema.json)
- [Training Engine](../src/training/README.md)
- [Inference README](../src/inference/README.md)
- [Kernel Tests](../tests/kernels/README.md)
