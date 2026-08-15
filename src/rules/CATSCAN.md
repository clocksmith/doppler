# CATSCAN: Declarative Rules

Component: `doppler.runtime-source.rules`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Resolve explicit data-only rule maps deterministically and fail when no permitted selection exists.

## Authority

- Owns rule loading, matching, validation, bundle generation inputs, and selection result semantics.
- Does not own rule policy content outside its domain or add JavaScript fallback decisions.

## Scope

- Converter, inference, kernel, loader, and tooling rule maps plus registry code.

## Contracts

- Input: Checked-in JSON rule maps and validated selection context.
- Output: Deterministic selected values through the [rule registry](rule-registry.js).

## Invariants

- Rule maps are JSON data with explicit match and value semantics.
- Model identity matching cannot broaden through substring heuristics.
- Missing or ambiguous required selection fails explicitly.

## Acceptance

- Rule bundle, execution-rule, capability-policy, and matcher tests pass.
- Evidence: [rule contract tests](../../tests/config).

## Non-goals

- Hiding conditional runtime policy inside a generic matcher.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
