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

*Last updated: 2026-02-23*

Note: Internal postmortems, planning notes, and capability matrices live in the private wrapper repo.
