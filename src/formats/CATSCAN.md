# CATSCAN: Artifact Formats

Component: `doppler.runtime-source.formats`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Parse and validate external and native artifact formats into explicit data contracts without executing models.

## Authority

- Owns format parsing, format validation, tokenizer-file normalization, and byte-layout interpretation.
- Does not own storage transport, runtime policy, GPU resources, or model support status.

## Scope

- GGUF, SafeTensors, TFLite, LiteRT, tokenizer, and RDRR format modules.

## Contracts

- Input: Artifact bytes and their published or repository-owned format specifications.
- Output: Validated format facts through [format exports](index.js).

## Invariants

- Parsing is deterministic and GPU-unaware.
- Malformed or unsupported layouts fail explicitly.
- Format facts remain separate from runtime-selected behavior.

## Acceptance

- Format parser, validation, and malformed-input tests pass.
- Evidence: [format tests](../../tests/formats).

## Non-goals

- Download policy, model execution, or conversion-time support claims.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
