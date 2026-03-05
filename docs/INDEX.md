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
- [Operations](operations.md) - troubleshooting and debug workflows.

## Testing and Benchmarks

- [Testing](testing.md) - testing index.
- [Testing Runbook](testing-runbook.md) - operational test execution.
- [Kernel Testing Design](kernel-testing-design.md) - kernel correctness design guidance.
- [Benchmark Methodology](benchmark-methodology.md) - fairness and claim publication policy.
- [Vendor Registry](../benchmarks/vendors/README.md) - cross-product benchmark contracts and tooling.
- [Release Matrix](release-matrix.md) - generated model/platform support snapshot.

## Training and Distill

- [Training Handbook](training-handbook.md) - canonical operations and gates.
- [Training Artifact Policy](training-artifact-policy.md)
- [Training Migrations](training-migrations.md)
- [Training Rollout Readiness](training-rollout-readiness.md)
- [Training Benchmark Publication](training-benchmark-publication.md)
- [Diffusion Expansion Lanes](diffusion-expansion-lanes.md)

## Style Guides

- [Style Guides](style/README.md)

## Specs and Source Readmes

- [Benchmark Schema](../benchmarks/benchmark-schema.json)
- [Training Engine](../src/training/README.md)
- [Inference README](../src/inference/README.md)
- [Kernel Tests](../tests/kernels/README.md)
