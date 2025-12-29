# DOPPLER Task Tracking

**All tasks are now tracked in the feature-log system.**

See:
- `feature-log/doppler/*.jsonl` - JSONL database of all features and tasks
- `/feature-log-query --status planned` - Query planned tasks
- `/feature-log-query --priority P0` - Query P0 tasks

---

## Technical Deep-Dives

For technical implementation details, see `docs/internals/`:
- [Quantization](internals/QUANTIZATION.md) - Q4K layouts, column-wise optimization
- [Matmul](internals/MATMUL.md) - Thread utilization, GEMV variants
- [Attention](internals/ATTENTION.md) - Decode kernel, barrier analysis
- [Fusion](internals/FUSION.md) - Kernel fusion opportunities
- [MoE](internals/MOE.md) - Expert paging, sparsity
- [Memory Tiers](internals/MEMORY_TIERS.md) - Tiered architecture

---

## Status Overview

For current operational status, test results, and recent fixes, see:
- [Postmortems](../postmortems/INDEX.md) (Issue history)
- [Architecture](ARCHITECTURE.md) (System design)
