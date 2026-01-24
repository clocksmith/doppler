# DOPPLER Docs Index

Quick index for DOPPLER documentation.

---

## Core Docs

- [Architecture](ARCHITECTURE.md) - System overview and pipeline.
- [Browser Capabilities](../../docs/BROWSER_CAPABILITIES.md) - Browser-only matrix, gaps, and references.
- [Config](CONFIG.md) - Kernel paths and error codes.
- [Formats](FORMATS.md) - RDRR and LoRA formats, adapter manifest.
- [Operations](OPERATIONS.md) - Troubleshooting, debug notes, perf investigations, results.
- [Benchmarking](BENCHMARKING.md) - Default bench workflow and baseline targets.
- [Testing](TESTING.md) - Testing strategy, kernel testing, known-good matrix.
- [Agent Intent Bundle Template](AGENT_INTENT_BUNDLE.md) - Review-ready change proposal checklist.
- [Intent Bundle Spec](../../docs/INTENT_BUNDLE.md) - Canonical schema and examples.
- [Roadmap](ROADMAP.md) - Vision, plans, competitive notes.
- [Traction](PERFORMANCE.md) - Performance metrics, VRAM constraints, browser matrix, failure modes.

---

## Style Guides

- [Style Guides](style/README.md) - General, JavaScript, WGSL, config, benchmark.

---

## Specs & Tooling

- [Benchmark Schema](spec/BENCHMARK_SCHEMA.json) - JSON schema for benchmark output.
- [Training Engine](../src/training/README.md) - Training primitives and runners.
- [Inference README](../src/inference/README.md) - Step-by-step inference flow.
- [Kernel Tests (Implemented)](../tests/kernels/README.md) - Kernel correctness and microbenchmarks.
- [Kernel Benchmarks](../tests/kernels/BENCHMARKS.md) - Baseline expectations and notes.

*Last updated: January 2026*

Note: Internal postmortems and deep-dive subsystem notes live in the private wrapper repo.
