# DOPPLER Docs Index

Primary documentation index.

## Start Here

- [Getting Started](getting-started.md) - canonical first-run workflow.
- [Setup](setup-instructions.md) - environment setup and troubleshooting.
- [CLI Quick Start](cli-quickstart.md) - command-shape reference.
- [Performance and Sizing](performance-sizing.md) - hardware tiers and planning guidance.

## Core Runtime Docs

- [Architecture](architecture.md) - system model and boundaries.
- [Pipeline Contract](pipeline-contract.md) - command-to-output runtime contract boundaries.
- [Config](config.md) - kernel paths, config behavior, and runtime contract notes.
- [Formats](formats.md) - format spec index (RDRR + LoRA).
- [RDRR Format](rdrr-format.md) - runtime artifact spec.
- [LoRA Format](lora-format.md) - adapter manifest spec.
- [Conversion Runtime Contract](conversion-runtime-contract.md) - conversion-static vs runtime-overridable ownership.
- [Model Promotion Playbook](model-promotion-playbook.md) - canonical sync workflow for repo metadata, external-volume RDRR artifacts, and Hugging Face hosting.
- [Registry Workflow](registry-workflow.md) - hosted catalog validation and Hugging Face publication workflow.
- [Operations](operations.md) - troubleshooting and debug workflows.

## Public API Docs

- [API Docs Index](api/index.md) - canonical public API navigation.
- [Root API](api/root.md) - app-facing `doppler` facade.
- [Advanced Root Exports](api/advanced-root-exports.md) - root-level loaders, adapters, and advanced exports.
- [Provider API](api/provider.md) - provider/singleton surface.
- [Generation API](api/generation.md) - lower-level text pipeline surface.
- [Diffusion API](api/diffusion.md) - diffusion/image pipeline surface.
- [Energy API](api/energy.md) - energy pipeline surface.
- [Tooling API](api/tooling.md) - command-runner and tooling surface.
- [Generated Export Inventory](api/reference/exports.md) - machine-derived export inventory from package entrypoints.

## Testing and Benchmarks

- [Testing](testing.md) - testing index.
- [Testing Runbook](testing-runbook.md) - operational test execution.
- [Kernel Testing Design](kernel-testing-design.md) - kernel correctness design guidance.
- [Benchmark Methodology](benchmark-methodology.md) - fairness and claim publication policy.
- [Vendor Registry](../benchmarks/vendors/README.md) - cross-product benchmark contracts and tooling.
- [Release Matrix](release-matrix.md) - generated model/platform support snapshot.
- [Release Notes](release-notes.md) - non-generated notes for tooling and workflow changes.

## Training and Distillation

- [Training Handbook](training-handbook.md) - canonical operator workflow, gates, and artifact contract.
- [Training Overview](training-overview.md) - concise map of operator vs legacy harness flows.
- [Training Operator Playbook](training-operator-playbook.md) - day-to-day sequence for `lora` and `distill`.
- [Training Artifact Policy](training-artifact-policy.md)
- [Training Governance](training-governance.md)
- [Training Migrations](training-migrations.md)
- [Training Rollout Readiness](training-rollout-readiness.md)
- [Training Benchmark Publication](training-benchmark-publication.md)
- [Distill Studio Ops](distill-studio-ops.md) - legacy compatibility helpers, not the primary operator path.
- [Diffusion Expansion Lanes](diffusion-expansion-lanes.md)

## Style Guides

- [Style Guides](style/README.md)

## Specs and Source Readmes

- [Benchmark Schema](../benchmarks/benchmark-schema.json)
- [Training Engine](../src/training/README.md)
- [Inference README](../src/inference/README.md)
- [Kernel Tests](../tests/kernels/README.md)
