# CATSCAN: Browser Demo

Component: `doppler.demo`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Give users one dependable browser surface for verified local model execution and inspectable evidence.

## Authority

- Owns the hosted demo experience, public API consumption, model lifetime UI, and evidence presentation.
- Does not own private runtime algorithms, model facts, or independent command semantics.

## Scope

- The live demo entrypoints, UI modules, assets, PWA shell, and demo receipts.

## Contracts

- Input: [Demo contract](README.md) and public package APIs.
- Output: User-visible generation, inspection views, and reproducible demo receipts.

## Invariants

- Live code uses public package boundaries.
- Observation tiers disclose when measurement changes execution.
- The demo never implies support beyond the support registry.

## Acceptance

- `npm run demo:reachability:check` and `npm run test:demo:contract` pass.
- Evidence: [demo contract tests](../tests/demo).

## Non-goals

- A second runtime implementation or a privileged bypass around package contracts.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
