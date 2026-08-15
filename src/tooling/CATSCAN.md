# CATSCAN: Shared Command Tooling

Component: `doppler.runtime-source.tooling`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Give browser, Node, and CLI adapters one normalized command contract and one evidence-aware execution vocabulary.

## Authority

- Owns command normalization, command contexts, shared runners, envelopes, diagnostics, calibration, and tooling evidence contracts.
- Does not own surface-specific presentation or permit adapters to redefine command semantics.

## Scope

- Shipped tooling APIs and shared browser/Node command infrastructure.

## Contracts

- Input: Normalized requests, [command rules](../rules/tooling), runtime profiles, artifacts, and observation policy.
- Output: Validated command envelopes, executions, diagnostics, and receipts through the [command API](command-api.js).

## Invariants

- Request intent is the sole active command authority.
- Browser and Node preserve shared semantics; unsupported capabilities fail closed.
- Evidence capture stays distinct from numerical runtime policy.

## Acceptance

- Command-surface, runner parity, workflow, and tooling tests pass.
- Evidence: [tooling tests](../../tests/tooling).

## Non-goals

- A repository-dev script bucket or a surface-specific alternate runtime.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
