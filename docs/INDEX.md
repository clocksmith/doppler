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
- [UL v1 (Practical)](training-ul-v1.md) - UL stage commands, artifacts, and claim boundaries.
- [Training Release Gates](training-release-gates.md) - Blocking CI/policy gates for training and UL claims.
- [Training Metrics Migration v1](training-metrics-migration-v1.md) - Objective-aware metrics schema migration notes.
- [Training Command Migration v1](training-command-migration-v1.md) - Training command payload schema/version migration notes.
- [UL Schema Changelog](training-ul-schema-changelog.md) - UL config schema version and change history.
- [Training Contract Invariants](training-contract-invariants.md) - Verify vs calibrate invariants and fail-closed rules.
- [Training Compatibility Matrix](training-contract-compatibility-matrix.md) - Browser/Node/CLI training field parity.
- [Training Lineage Requirements](training-lineage-requirements.md) - Required hash-linked lineage per artifact type.
- [Training Artifact Conventions](training-artifact-conventions.md) - Naming/versioning policy for training and UL artifacts.
- [Training Hash Policy](training-hash-policy.md) - Canonical hashing policy for linkage and claims.
- [Training Timestamp Policy](training-deterministic-timestamp-policy.md) - Deterministic hash behavior vs volatile timestamps.
- [Distill Studio MVP](distill-studio-mvp.md) - MVP contract surface and operator commands.
- [Distill Studio Rollout Checklist](distill-studio-rollout-checklist.md) - Staged rollout controls.
- [Distill Studio Reliability Dashboard](distill-studio-reliability-dashboard.md) - Operator dashboard signals.
- [Distill Studio Incident Playbook](distill-studio-incident-playbook.md) - Incident response flow.
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

*Last updated: 2026-03-01*

Note: Internal postmortems, planning notes, and capability matrices live in the private wrapper repo.
