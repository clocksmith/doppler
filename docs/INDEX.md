# DOPPLER Docs Index

Quick index for DOPPLER documentation.

---

## Vision & Roadmap

**Start here:** [VISION.md](VISION.md) - Capability thesis and phased roadmap overview.

| Phase | Goal | Status | Roadmap |
|-------|------|--------|---------|
| 1 | Performance Parity | In Progress | [PHASE_1_PERFORMANCE.md](roadmap/PHASE_1_PERFORMANCE.md) |
| 2 | MoE Efficiency | Partial | [PHASE_2_MOE.md](roadmap/PHASE_2_MOE.md) |
| 3 | Scale Beyond WebLLM | Planned | [PHASE_3_SCALE.md](roadmap/PHASE_3_SCALE.md) |
| 4 | P2P Distribution | Design | [PHASE_4_P2P.md](roadmap/PHASE_4_P2P.md) |
| 5 | Evolution | Design | [PHASE_5_EVOLUTION.md](roadmap/PHASE_5_EVOLUTION.md) |

**Infrastructure:**

| Feature | Status | Roadmap |
|---------|--------|---------|
| YAML Kernel Config | Planned | [KERNEL_CONFIG_SYSTEM.md](roadmap/KERNEL_CONFIG_SYSTEM.md) |

---

## Core Docs

- [Architecture](ARCHITECTURE.md) - High-level module layout and subsystem responsibilities.
- [Execution Pipeline](EXECUTION_PIPELINE.md) - Kernel-by-kernel inference walkthrough and fusion analysis.
- [Glossary](GLOSSARY.md) - Terms and definitions used across DOPPLER.
- [Troubleshooting Guide](DOPPLER-TROUBLESHOOTING.md) - Comprehensive debugging strategies for inference issues.
- [Inference README](../inference/README.md) - Step-by-step inference flow (init, load, prefill, decode).
- [Hardware Compatibility](HARDWARE_COMPATIBILITY.md) - Browser and GPU support notes.

---

## Reference Docs

| Document | Content |
|----------|---------|
| [Model Support](plans/MODEL_SUPPORT.md) | Model compatibility matrix |
| [DRY Refactoring Plan](roadmap/DRY_REFACTORING.md) | Consolidated DRY/refactor plan and status |
| [Competitive Analysis](analysis/COMPETITIVE.md) | WebLLM, WeInfer, Transformers.js comparison |
| [RDRR Format](spec/RDRR_FORMAT.md) | Model format specification |

---

## Specs & Testing

- [Benchmark Harness](spec/BENCHMARK_HARNESS.md) - Standardized benchmarking spec and JSON output schema.
- [Kernel Testing](spec/KERNEL_TESTING.md) - WGSL unit tests and pipeline segment tests.
- [Kernel Tests (Implemented)](../kernel-tests/TODO.md) - Kernel correctness and microbenchmark tracking.
- [Kernel Benchmarks](../kernel-tests/BENCHMARKS.md) - Baseline expectations and benchmark notes.

---

## Results

- [Test Results](TEST_RESULTS.md) - Benchmark and validation logs by session.

---

## Postmortems

Notes and incident writeups live in `docs/postmortems/`.

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
