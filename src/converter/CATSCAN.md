# CATSCAN: Artifact Conversion

Component: `doppler.runtime-source.converter`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Materialize source checkpoints into explicit, reproducible runtime artifacts without inventing runtime policy later.

## Authority

- Owns source parsing coordination, tensor transformation, quantization, sharding, and manifest materialization.
- Does not own runtime overrides, model support claims, or post-conversion mutation of artifact facts.

## Scope

- Shared conversion plans, quantizers, parsers, tokenizer copying, and shard packing.

## Contracts

- Input: Source checkpoints and [conversion/runtime ownership contract](../../docs/conversion-runtime-contract.md).
- Output: Artifacts conforming to the [RDRR charter](../formats/rdrr/CATSCAN.md).

## Invariants

- Conversion-owned facts are explicit and reproducible.
- Quantization layout matches its declared format exactly.
- Unsupported source layouts fail before emitting a misleading artifact.

## Acceptance

- Converter, quantizer, and manifest contract tests pass.
- Evidence: [converter tests](../../tests/converter).

## Non-goals

- Runtime family detection or declaring a converted artifact product-supported.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
