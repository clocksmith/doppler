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
- [Test harness](test-harness.js): catalog discovery, browser options, experimental diagnostics.
- [Capsule Forge command](model-capsule-forge.js): files and signing around the [compiler](../converter/CATSCAN.md).
- [Model onboarding](model-onboarding.js) coordinates pinned source assessment,
  lineage, conversion, independent comparison, physical qualification, and signed
  Capsule execution resumably. Retains immutable outputs and failures.
  Preparation cannot establish inference; onboarding cannot publish or approve application upgrades.

## Contracts

- Input: Normalized requests, [command rules](../rules/tooling), runtime profiles, artifacts, and observation policy.
- Output: Validated command envelopes, executions, diagnostics, and receipts through the [command API](command-api.js).

## Invariants

- Request intent is the sole active command authority.
- Browser and Node preserve shared semantics; unsupported capabilities fail closed.
- Evidence capture stays distinct from numerical runtime policy.
- Node provider selection and reversible global installation are owned here;
  standard WebGPU execution must not require Doe's package or contract loader.
  Explicit external provider contracts remain optional interoperability.

## Acceptance

- Command-surface, runner parity, workflow, and tooling tests pass.
- Evidence: [tooling tests](../../tests/tooling).

## Non-goals

- Repository-dev scripts or alternate runtimes.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
