# Benchmark Methodology

Canonical policy for benchmark fairness and claim publication.

## Claim requirements

Every claim must include:
- workload id
- engine/version
- cache/load mode
- command used
- artifact path(s)

## Canonical timing contract

Required metrics:
- `modelLoadMs`
- `firstTokenMs`
- `firstResponseMs`
- `prefillMs`
- `decodeMs`
- `totalRunMs`
- `decodeTokensPerSec`
- `prefillTokensPerSec`
- `decodeMsPerTokenP50`
- `decodeMsPerTokenP95`
- `decodeMsPerTokenP99`

## Fair comparison rules

- same model identity and quantization class, or explicit equivalence map
- same prompt/workload settings
- same sampling settings
- same hardware class/browser family for direct claims
- explicit disclosure of any deviation

## Tooling contract

Use the vendor registry toolchain:
- `benchmarks/vendors/registry.json`
- `benchmarks/vendors/workloads.json`
- `benchmarks/vendors/harnesses/*.json`
- `tools/vendor-bench.js`
- `tools/compare-engines.js`

## Standard commands

```bash
node tools/vendor-bench.js validate
node tools/vendor-bench.js capabilities
node tools/vendor-bench.js gap --base doppler --target transformersjs
node tools/compare-engines.js --mode all
```

## Related

- Vendor registry and CLI details: [../benchmarks/vendors/README.md](../benchmarks/vendors/README.md)
- Style-specific implementation notes: [style/benchmark-style-guide.md](style/benchmark-style-guide.md)
