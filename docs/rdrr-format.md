# RDRR Format

Canonical specification for Doppler runtime model artifacts.

## Scope

RDRR is the runtime contract consumed by Doppler loading and inference paths.

Related specs and contracts:
- [lora-format.md](lora-format.md) for adapter manifests
- [conversion-runtime-contract.md](conversion-runtime-contract.md) for conversion-vs-runtime ownership
- [getting-started.md](getting-started.md) for first-run convert/verify flow

## Core structure

An RDRR artifact set contains:
- `manifest.json`
- `shard_*.bin`
- tokenizer assets where required

## Goals

- deterministic loading contract
- auditable manifest-first execution policy
- explicit quantization and inference defaults

## Required manifest fields (v1)

At minimum, manifests must include:
- model identity
- architecture/modelType
- tensor map/shard entries
- quantization metadata when quantized
- inference defaults needed for runtime selection

## Invariants

- Manifest is validated before runtime dispatch.
- Unresolved execution choices fail closed.
- Runtime behavior is derived from manifest + runtime config merge.
- If `manifest.inference.schema == "doppler.execution/v0"`, the manifest must
  include explicit `sessionDefaults` and `execution.steps`:
  - `sessionDefaults.compute.defaults.{activationDtype,mathDtype,accumDtype,outputDtype}`
  - `sessionDefaults.compute.kernelProfiles`
  - `sessionDefaults.kvcache` (nullable, but explicit)
  - `sessionDefaults.decodeLoop` (nullable, but explicit)
  - pinned `kernelRef` for each non-cast execution step

## Related implementation

- `src/formats/rdrr/parsing.js`
- `src/formats/rdrr/validation.js`
- `src/formats/rdrr/manifest.js`
- `src/storage/shard-manager.js`
- `src/storage/downloader.js`

## Kernel override note

For override and compatibility policy, use the canonical section in [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility).
