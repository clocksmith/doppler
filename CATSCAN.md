# CATSCAN: Doppler Repository

Component: `doppler`

Parent: none

## Target

Deliver inspectable, evidence-backed local model execution for JavaScript applications.

## Authority

- Owns Doppler's repository mission, product boundaries, artifact/runtime contract, and evidence standard.
- Does not own application intent, agent policy, or unsupported capability claims.

## Scope

- The shipped package, browser demo, model catalog, qualification machinery, documentation, and retained evidence.

## Contracts

- Input: [Durable goals](docs/goals.md) and [subsystem support policy](src/config/support-tiers/subsystems.json).
- Output: [Package surfaces](package.json), qualified artifacts, runtime behavior, and auditable receipts.

## Invariants

- Supported behavior is stronger than available code and must remain explicitly scoped.
- Artifact, tokenizer, graph, kernel, provider, and runtime-policy identity remain inspectable.
- Unsupported or unresolved execution choices fail closed.

## Acceptance

- `npm run check:green` passes from a clean checkout.
- Evidence: [goal completion matrix](src/config/goal-completion-matrix.json).

## Non-goals

- Application orchestration, universal model coverage, and unqualified performance claims.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
