# Benchmark Protocol

Referenced by: `doppler-bench`, `doppler-perf`

## Vendor Benchmark Registry

Cross-product benchmark tracking lives under `benchmarks/vendors/`.

- Registry: `benchmarks/vendors/registry.json`
- Workloads: `benchmarks/vendors/workloads.json`
- Capability matrix: `benchmarks/vendors/capabilities.json`
- Harness definitions: `benchmarks/vendors/harnesses/*.json`
- Normalized outputs: `benchmarks/vendors/results/`
- CLI: `tools/vendor-bench.js`

## Registry Commands

```bash
node tools/vendor-bench.js validate
node tools/vendor-bench.js capabilities
node tools/vendor-bench.js gap --base doppler --target transformersjs
```

## Update Checklist

When harness/profiling behavior changes (Doppler or vendors), update:
1. Harness definition in `benchmarks/vendors/harnesses/`
2. Capability matrix in `benchmarks/vendors/capabilities.json`
3. Docs in `benchmarks/vendors/README.md`

## Claimable Compare Checklist

Before promoting a benchmark number into README copy, a chart, or any other claimable artifact:

1. Confirm the active artifact source is the one you intend to measure.
   - Local manifests and published HF artifacts can drift.
   - Do not assume a local fix is reflected in a published compare lane until the compare artifact proves it.
2. Treat correctness as a gate, not a footnote.
   - If compare output carries a correctness mismatch, do not promote the perf number as claimable evidence.
   - Fix prompt parity first, then decode parity, then performance.
3. Benchmark the real lane, not a debug surrogate.
   - Do not use `f32/http` or investigation profiles as performance evidence for a warmed browser/OPFS compare lane.
4. Record the exact compare artifact path and command.
   - Human-facing claims should be traceable back to one saved JSON artifact and one reproducible command.
5. When mixing legacy `warm` fixtures with newer `compute/parity` fixtures for visualization, make that normalization explicit in the chart contract and lock it with a test.

## Parity vs Throughput Lanes

Keep compare evidence in two explicit buckets:

1. `parity lane`
   - fairness-managed and claimable when correctness is clean
   - allowed to force single-token decode or disable engine-specific speculation when that is needed for apples-to-apples semantics
   - must be saved as a committed compare fixture before it is treated as benchmark truth
2. `throughput lane`
   - tuning evidence for Doppler-favorable decode cadence or phase-specific experiments
   - not automatically claimable, even when it is faster

For performance work, localize the phase before patching:
- prefill and decode are separate investigations
- dump resolved prefill/decode kernel path and hot weight/materialization dtype separately
- for decode, record `decodeMode`, `batchGuardReason`, and whether the wall is in `decodeRecordMs` or submit/readback wait before changing kernels
