# Doppler.js

Doppler is a local inference engine for browser-hosted and on-device AI workloads.
It provides explicit load-path and kernel-path control, reproducible benchmarking, and auditable kernel execution tracing.

## Evidence

Lower is better, comparing per-phase latency by workload.

![Phase-latency comparison on one workload across models](benchmarks/vendors/results/compare_1b_multi-workload_favorable_phases.svg)

Snapshot artifacts:
- [compare_20260303T175640.json](benchmarks/vendors/results/compare_20260303T175640.json)
- [compare_20260303T210150.json](benchmarks/vendors/results/compare_20260303T210150.json)

## Start here

- Canonical first-run guide: [docs/getting-started.md](docs/getting-started.md)
- Setup and troubleshooting: [docs/setup-instructions.md](docs/setup-instructions.md)
- Sizing and performance expectations: [docs/performance-sizing.md](docs/performance-sizing.md)

## Core docs

- Architecture: [docs/architecture.md](docs/architecture.md)
- Pipeline contract: [docs/pipeline-contract.md](docs/pipeline-contract.md)
- Config: [docs/config.md](docs/config.md)
- Formats index: [docs/formats.md](docs/formats.md)
- Operations: [docs/operations.md](docs/operations.md)
- Testing index: [docs/testing.md](docs/testing.md)
- Training handbook: [docs/training-handbook.md](docs/training-handbook.md)
- Benchmark methodology: [docs/benchmark-methodology.md](docs/benchmark-methodology.md)

## License

Apache License 2.0 (`Apache-2.0`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
