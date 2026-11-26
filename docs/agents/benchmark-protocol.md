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

Claim requirements, fairness rules and the pre-promotion checklist are canonical in
[Benchmark Methodology](../benchmark-methodology.md#claimable-compare-checklist).

## Parity vs Throughput Lanes

See [Benchmark Methodology](../benchmark-methodology.md#parity-vs-throughput-lanes).

## Localize the Phase Before Patching

For performance work:
- prefill and decode are separate investigations
- dump resolved prefill/decode kernel path and hot weight/materialization dtype separately
- for decode, record `decodeMode`, `batchGuardReason`, and whether the wall is in `decodeRecordMs` or submit/readback wait before changing kernels
- classify the wall before any kernel edit: GPU compute versus submit/readback/orchestration; do not assume a math-kernel bug when timing shows orchestration dominating
