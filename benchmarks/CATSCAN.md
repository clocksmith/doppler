# CATSCAN: Benchmark System

Component: `doppler.benchmarks`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Produce reproducible measurements whose workload, identity, timing scope, and claim status are explicit.

## Authority

- Owns benchmark workload contracts, measurement orchestration, normalization, and retained results.
- Does not own runtime correctness, model support promotion, or application-level quality policy.

## Scope

- Doppler benchmarks, cross-engine comparisons, runners, fixtures, and result artifacts.

## Contracts

- Input: [Benchmark methodology](../docs/benchmark-methodology.md) and exact runtime/artifact identities.
- Output: Normalized measurements and comparison evidence under [vendor results](vendors/results).

## Invariants

- Compared lanes perform declared comparable work.
- Correctness and timing scope are visible before performance claims are allowed.
- Failed, diagnostic, and throughput-only runs cannot silently become claimable.

## Acceptance

- Benchmark schemas and committed fixtures validate through the declared package checks.
- Evidence: [vendor benchmark fixtures](vendors/fixtures).

## Non-goals

- Declaring broad product support from one machine, shape, or isolated measurement.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
