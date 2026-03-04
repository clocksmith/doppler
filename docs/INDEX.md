# DOPPLER Docs Index

Quick index for DOPPLER documentation.

---

## Core Docs

- [Architecture](architecture.md) - System overview and pipeline.
- [Pipeline Contract](pipeline-contract.md) - Boundary-by-boundary flow from command entry to output.
- [Config](config.md) - Kernel paths and error codes.
- [Formats](formats.md) - RDRR and LoRA formats, adapter manifest.
- [Operations](operations.md) - Troubleshooting, debug notes, perf investigations, results.
- [Testing](testing.md) - Testing strategy, kernel testing, known-good matrix.
- [Training Overview](training-overview.md) - Verify/calibrate scope, stage support, and release expectations.
- [Training Artifact Policy](training-artifact-policy.md) - Naming, hashing, lineage, and deterministic timestamp policy.
- [Training Migrations](training-migrations.md) - Command/schema/metrics migration baseline and guidance.
- [Training Claim Traceability](training-claim-traceability.md) - Deterministic workload hashes and report-id mapping policy.
- [Training Operator Playbook](training-operator-playbook.md) - Operational command sequence for gates, distill MVP, and incidents.
- [Training Rollout Readiness](training-rollout-readiness.md) - Release/rollout checklist for contract and traceability readiness.
- [Training Benchmark Publication](training-benchmark-publication.md) - Required artifact/process for benchmark claim publication.
- [Diffusion Expansion Lanes](diffusion-expansion-lanes.md) - Started backlog for larger targets, quantized diffusion, and platform rollout.
- [Distill Studio Ops](distill-studio-ops.md) - Distill Studio contract surface, rollout, reliability, incident response.
- [Release Matrix](release-matrix.md) - Generated model/platform support snapshot tied to vendor benchmark configs.
- [Onboarding Tooling](onboarding-tooling.md) - Check and scaffold workflows for conversion, kernels, and behavior presets.

---

## Style Guides

- [Style Guides](style/README.md) - General, JavaScript, WGSL, config, benchmark.

---

## Specs & Tooling

- [Benchmark Schema](../benchmarks/benchmark-schema.json) - JSON schema for benchmark output.
- [Vendor Registry](../benchmarks/vendors/README.md) - Cross-product benchmark targets, harnesses, and normalized results.
- [Training Engine](../src/training/README.md) - Training primitives and runners.
- [Inference README](../src/inference/README.md) - Step-by-step inference flow.
- [Kernel Tests (Implemented)](../tests/kernels/README.md) - Kernel correctness and microbenchmarks.
- [Kernel Benchmarks](../tests/kernels/benchmarks.md) - Baseline expectations and notes.

*Last updated: 2026-03-04*

Note: Internal postmortems, planning notes, and capability matrices live in the private wrapper repo.
