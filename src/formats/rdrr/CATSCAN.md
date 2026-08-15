# CATSCAN: RDRR Format

Component: `doppler.runtime-source.formats.rdrr`

Parent: [Artifact Formats](../CATSCAN.md)

## Target

Define and validate Doppler's canonical manifest-first runtime artifact contract.

## Authority

- Owns RDRR manifest parsing, validation, tensor/shard references, and additive integrity metadata shape.
- Does not own shard transport, artifact promotion, runtime overrides, or direct-source artifact semantics.

## Scope

- RDRR parsing, validation, manifest helpers, and distributed format extensions.

## Contracts

- Input: [RDRR specification](../../../docs/rdrr-format.md) and artifact manifest data.
- Output: Validated RDRR manifest contracts through [RDRR exports](index.js).

## Invariants

- Manifest validation precedes runtime dispatch.
- Session and execution choices required by the format are explicit.
- Integrity extensions never silently rewrite canonical artifact identity.

## Acceptance

- RDRR parsing, validation, shard, and integrity tests pass.
- Evidence: [RDRR format tests](../../../tests/formats).

## Non-goals

- Treating direct-source bundles as synthetic RDRR or resolving missing shards implicitly.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
