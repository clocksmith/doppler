# CATSCAN: Model Catalog

Component: `doppler.model-catalog`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Govern the exact model identities and lifecycle facts that Doppler can resolve, qualify, and expose.

## Authority

- Owns catalog metadata, logical model records, artifact references, classification, and promotion lifecycle state.
- Does not own runtime execution semantics, source checkpoint behavior, or claims unsupported by receipts.

## Scope

- Canonical model catalog data, target inventories, taxonomy, and adapter catalog metadata.

## Contracts

- Input: Artifact identities, qualification receipts, hosted locations, and [model taxonomy](model-type-taxonomy.json).
- Output: [Canonical model catalog](catalog.json) and generated public registry inputs.

## Invariants

- Logical identity and resolved artifact identity remain distinct.
- Published availability and verified support are separate states.
- Runtime mirrors cannot silently rewrite catalog facts.

## Acceptance

- `npm run artifact:contract:check` and catalog checks pass.
- Evidence: [catalog contract tests](../tests/config).

## Non-goals

- Maximizing catalog size or treating conversion existence as product support.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
