# CATSCAN: Model Host Composition

Component: `doppler.runtime-source.client.model-host`

Parent: [Client API](../CATSCAN.md)

## Target

Compose model acquisition and application handles above the injected Capsule execution core.

## Authority

- Owns convenience loading, device initialization coordination, model caching, input formatting, and model-handle evidence construction.
- Does not own Capsule semantics, qualification, application trust, or GPU computation.

## Scope

- [Host service](index.js), [model handles](model-session.js), and [evidence construction](model-evidence.js).
- Previous `client/runtime/index` and `model-session` paths remain compatibility facades.

## Contracts

- Input: [Capsule core](../runtime/composition-root.js), declared source/loading contracts, application policy, and pipeline observations.
- Output: [Public host interface](../capsule-host.js), model handles, and attributable evidence.

## Invariants

- Host composition may depend on Capsule execution; Capsule execution cannot import this host layer.
- Evidence construction cannot acquire GPU resources or invoke a model pipeline.
- Forwarding entry points expose the same Capsule contract without former product-name aliases.
- Loading and cancellation failures preserve cleanup and the original failure.
- Receipted token constraints require a declared content identity bound into the generation configuration hash.

## Acceptance

- `npm run source:architecture:check` and connected client/Capsule tests pass.
- Evidence: [host boundary regression](../../../tests/client/model-host-boundaries.test.js).

## Non-goals

- Reimplementing inference, automatic upgrades, and changing executable identity during loading.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
