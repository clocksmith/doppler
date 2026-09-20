# CATSCAN: Doppler Repository

Component: `doppler`

Parent: none

## Target

Make local AI an ordinary JavaScript dependency. Prioritize copy-and-run local search; long-term success is independent adoption.

## Authority

- Owns mission, product boundaries, artifact/runtime contracts, and evidence standards.
- Owns reference applications; their controllers and search policies stay outside runtime.
- Does not own consumer intent, agent policy, or unsupported claims.

## Scope

- Package, demo, examples, catalog, qualification, documentation, and evidence.

## Contracts

- Input: [Strategic goals](GOALS.md)
- Input: [System intent and invariants](INTENT.md)
- Input: [Subsystem support policy](src/config/support-tiers/subsystems.json).
- Output: [Package surfaces](package.json), qualified artifacts, execution, and receipts.

## Invariants

- Support requires scoped evidence, not available code.
- Artifact, tokenizer, graph, kernel, provider, and policy identity remain inspectable.
- Unsupported or unresolved execution choices fail closed.
- Standalone execution is independent; redistribution and delegation require authority.
- Free adoption counts; revenue, acquisition, Doe, Poolday, and Reploid are not gates.
- Network artifact, execution, and history-improvement proof remains separate.
- Forge prepares; Capsules bind signed implementations; Runtime executes without silent changes.

## Acceptance

- `npm run check:green` passes from a clean checkout.
- Evidence: [goal completion matrix](src/config/goal-completion-matrix.json).
- Meet [clean-environment search acceptance](docs/goals.md#next-product-increment-copy-and-run-local-search); reference apps are not independent adoption or promotion.

## Non-goals

- Runtime application orchestration, duplicate search engines, universal model coverage, unqualified performance claims.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
