# CATSCAN: Repository Tooling

Component: `doppler.repository-tooling`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Automate repository governance, generation, qualification, and operator workflows without entering the shipped runtime.

## Authority

- Owns development checks, generated-file synchronization, policy validation, qualification scripts, and repository reports.
- Does not own shipped command semantics except by validating their declared contracts.

## Scope

- Internal scripts and policies under `tools/`, excluding the explicitly packaged converter entrypoint.

## Contracts

- Input: Source files, policy JSON, schemas, registries, fixtures, and retained evidence.
- Output: Deterministic checks, generated mirrors, qualification artifacts, and actionable failures.

## Invariants

- Recurring drift classes gain check mode and a single policy source.
- Generated files are reproducible and stale state fails clearly.
- Tools do not silently mutate production behavior while checking it.

## Acceptance

- Repository policy, generation, and default check chains pass.
- Evidence: [CATSCAN validator](sync-catscan-index.js).

## Non-goals

- Shipping repository automation as runtime product code or replacing semantic review.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
