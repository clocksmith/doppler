# CATSCAN: Acceptance Tests

Component: `doppler.tests`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Provide executable evidence that contracts hold on success, failure, lifecycle, parity, and regression paths.

## Authority

- Owns deterministic test fixtures, unit and integration assertions, browser harness tests, and regression coverage.
- Does not own product claims beyond what a test actually exercises or replace retained production receipts.

## Scope

- Repository test suites, references, fixtures, and test-only helpers.

## Contracts

- Input: Source contracts, policy assets, frozen fixtures, and declared test environments.
- Output: Pass/fail evidence and localized regression boundaries.

## Invariants

- Tests state the contract they protect and include failure paths where relevant.
- Mocked execution is never presented as hardware evidence.
- Pending capability stays visibly pending rather than weakening an assertion.

## Acceptance

- `npm run test:unit` and applicable focused suites pass deterministically.
- Evidence: [integration test suite](integration).

## Non-goals

- Encoding mutable implementation details or converting one test pass into broad support.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
