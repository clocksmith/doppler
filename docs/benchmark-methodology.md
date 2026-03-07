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

Compare-artifact note:
- Cross-engine compare artifacts use apples-to-apples prompt metrics only:
  `firstTokenMs` and `promptTokensPerSecToFirstToken`.
- Raw engine payloads may still include `prefillMs` and `prefillTokensPerSec`, but those stay engine-local unless the semantics are proven identical.
- Capability matrices and release-matrix compare summaries must be updated when the compare metric contract changes.

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
