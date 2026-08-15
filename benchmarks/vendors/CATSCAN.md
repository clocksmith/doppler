# CATSCAN: Vendor Comparisons

Component: `doppler.benchmarks.vendors`

Parent: [Benchmark System](../CATSCAN.md)

## Target

Compare Doppler and eligible incumbent runtimes under shared, identity-bound evidence contracts.

## Authority

- Owns vendor registries, shared workloads, fairness policy, normalized compare records, and claim matrices.
- Does not own vendor implementations or permit format and output differences to remain hidden.

## Scope

- Cross-product harness definitions, capability declarations, results, and release-facing comparison state.

## Contracts

- Input: [Shared workloads](workloads.json), [benchmark policy](benchmark-policy.json), and provider outputs.
- Output: [Release matrix](release-matrix.json) and traceable comparison artifacts.

## Invariants

- Shared workload fields remain separate from engine overlays.
- Model, artifact, backend, output-policy, and timing identities remain explicit.
- A blocked or mismatched lane remains non-claimable.

## Acceptance

- `npm run bench:vendors:validate` accepts registries, schemas, and referenced evidence.
- Evidence: [committed comparison fixtures](fixtures).

## Non-goals

- Manufacturing a runtime win by changing workload, format, cache, or correctness rules invisibly.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
