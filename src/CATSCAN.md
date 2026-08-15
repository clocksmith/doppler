# CATSCAN: Shipped Source

Component: `doppler.runtime-source`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Implement Doppler's shipped JavaScript and WGSL contracts with explicit policy resolution and deterministic ownership.

## Authority

- Owns package runtime behavior, public source entrypoints, execution mechanisms, and source-level contracts.
- Does not own repository-only automation, benchmark claim policy, or application orchestration.

## Scope

- Code and checked-in runtime assets shipped from `src/`.

## Contracts

- Input: [Architecture](../docs/architecture.md), manifests, runtime policy, and command requests.
- Output: [Root package facade](index.js), tooling slices, and executed model results with receipts.

## Invariants

- JSON owns behavior policy, JavaScript owns orchestration, and WGSL owns math.
- Public facades delegate rather than acquire hidden execution semantics.
- Missing required runtime choices fail before dispatch.

## Acceptance

- Source architecture, browser imports, type checks, and unit tests pass.
- Evidence: [integration tests](../tests/integration).

## Non-goals

- Repository governance scripts, product copy, and undeclared fallback behavior.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
