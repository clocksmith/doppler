# DOPPLER Docs Index

Quick index for DOPPLER documentation.

---

## Core Docs

- [Architecture](architecture.md) - System overview and pipeline.
- [Config](config.md) - Kernel paths and error codes.
- [Formats](formats.md) - RDRR and LoRA formats, adapter manifest.
- [Operations](operations.md) - Troubleshooting, debug notes, perf investigations, results.
- [Testing](testing.md) - Testing strategy, kernel testing, known-good matrix.
- [Performance](performance.md) - Performance metrics, benchmarking workflow, VRAM constraints, browser matrix, failure modes.

---

## Style Guides

- [Style Guides](style/README.md) - General, JavaScript, WGSL, config, benchmark.

---

## Specs & Tooling

- [Benchmark Schema](BENCHMARK_SCHEMA.json) - JSON schema for benchmark output.
- [Training Engine](../src/training/README.md) - Training primitives and runners.
- [Inference README](../src/inference/README.md) - Step-by-step inference flow.
- [Kernel Tests (Implemented)](../tests/kernels/README.md) - Kernel correctness and microbenchmarks.
- [Kernel Benchmarks](../tests/kernels/benchmarks.md) - Baseline expectations and notes.

*Last updated: January 2026*

Note: Internal postmortems, planning notes, and capability matrices live in the private wrapper repo.
