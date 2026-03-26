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
