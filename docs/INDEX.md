# DOPPLER Docs Index

Quick index for DOPPLER documentation.

---

## Vision & Task Tracking

**Start here:** [VISION.md](VISION.md) - Capability thesis and phased roadmap overview.

**Task tracking:** All tasks are tracked in the feature-log system:
- `feature-log/doppler/*.jsonl` - JSONL database of all features and tasks
- `/feature-log-query --status planned` - Query planned tasks
- `/feature-log-query --priority P0` - Query P0 tasks

---

## Technical Internals

Deep-dives on kernel implementations and optimization strategies:

| Topic | Content |
|-------|---------|
| [Quantization](internals/QUANTIZATION.md) | Q4K layouts, column-wise optimization, GPU coalescing |
| [Matmul](internals/MATMUL.md) | Thread utilization, GEMV variants, LM head optimization |
| [Attention](internals/ATTENTION.md) | Decode kernel, barrier analysis, subgroup optimization |
| [Fusion](internals/FUSION.md) | Kernel fusion opportunities, optimization matrix |
| [MoE](internals/MOE.md) | Expert paging, sparsity exploitation, routing |
| [Memory Tiers](internals/MEMORY_TIERS.md) | Tiered architecture, tensor parallelism, P2P mesh |

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
| [Competitive Analysis](design/COMPETITIVE_ANALYSIS.md) | WebLLM, WeInfer, Transformers.js comparison |
| [RDRR Format](design/RDRR_FORMAT.md) | Model format specification |

---

## Specs & Testing

- [Benchmark Harness](design/BENCHMARK_HARNESS.md) - Standardized benchmarking spec and JSON output schema.
- [Kernel Testing](design/KERNEL_TESTING.md) - WGSL unit tests and pipeline segment tests.
- [Kernel Tests (Implemented)](../kernel-tests/TODO.md) - Kernel correctness and microbenchmark tracking.
- [Kernel Benchmarks](../kernel-tests/BENCHMARKS.md) - Baseline expectations and benchmark notes.

---

## Results

- [Test Results](TEST_RESULTS.md) - Benchmark and validation logs by session.

---

## Postmortems

Notes and incident writeups live in `docs/postmortems/`.

*Last updated: December 2025*
