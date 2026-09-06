# CATSCAN: Forge and Artifact Conversion

Component: `doppler.runtime-source.converter`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Compile source facts into ModelIR, qualified TargetPlans, and immutable Packs; materialize reproducible artifacts without inventing runtime policy later.

## Authority

- Owns source interpretation, semantic lowering, candidate evaluation, qualification construction, Pack construction, tensor transformation, and artifact materialization.
- Does not own runtime overrides, model support claims, or post-conversion mutation of artifact facts.

## Scope

- [Forge stages](forge-stages.js), source-fact validation, candidate evaluation, quantizers, tokenizer copying, and shard packing.
- [Command orchestration](../tooling/model-pack-forge.js) supplies file and signing inputs; compiler algorithms remain here.

## Contracts

- Input: Source checkpoints and [conversion/runtime ownership contract](../../docs/conversion-runtime-contract.md).
- Output: ModelIR, TargetPlans, Packs, evaluation receipts, and [RDRR artifacts](../formats/rdrr/CATSCAN.md).

## Invariants

- Conversion-owned facts are explicit and reproducible.
- Quantization layout matches its declared format exactly.
- Unsupported source layouts fail before emitting a misleading artifact.
- Candidate evaluation retains adverse observations and cannot substitute for qualification or application acceptance.

## Acceptance

- Converter, quantizer, and manifest contract tests pass.
- Evidence: [converter tests](../../tests/converter).

## Non-goals

- Runtime family detection or declaring a converted artifact product-supported.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
