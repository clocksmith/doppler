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

Registry and runner behavior are canonical in
[../benchmarks/vendors/README.md](../benchmarks/vendors/README.md).
Implementation conventions are in
[style/benchmark-style-guide.md](style/benchmark-style-guide.md).
