# CATSCAN: Model Source Contracts

Component: `doppler.runtime-source.model-contracts`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Expose narrow model-family source contracts without reintroducing runtime family detection or catalog authority.

## Authority

- Owns model-facing source modules whose semantics do not belong to a narrower runtime subsystem.
- Does not own canonical model catalog facts, conversion defaults, or hidden execution selection.

## Scope

- Shared family and workload source contracts under `src/models/`.

## Contracts

- Input: Explicit manifests, model configuration, and declared workload contracts.
- Output: Narrow reusable model modules such as [family contracts](family.js).

## Invariants

- Model names never substitute for manifest-owned execution facts.
- Source modules remain narrower than catalog, config, and pipeline authorities.
- New family behavior enters through explicit conversion and runtime contracts.

## Acceptance

- Model-family configuration and inference contract tests pass.
- Evidence: [model configuration tests](../../tests/config).

## Non-goals

- A second model registry or a fallback table keyed by model name.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
