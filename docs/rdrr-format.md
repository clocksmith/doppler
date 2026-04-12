# RDRR Format

Canonical specification for Doppler runtime model artifacts.

## Scope

RDRR is the runtime contract consumed by Doppler loading and inference paths.

Related specs and contracts:
- [lora-format.md](lora-format.md) for adapter manifests
- [conversion-runtime-contract.md](conversion-runtime-contract.md) for conversion-vs-runtime ownership
- [getting-started.md](getting-started.md) for first-run convert/verify flow

Non-RDRR note:
- Materialized direct-source artifacts use `manifest.json` plus raw SafeTensors/GGUF assets and declare `metadata.sourceRuntime.mode="direct-source"`. They are not RDRR shards and use their own digest/path contract.
- Runtime artifact loading for both `rdrr` and persisted direct-source manifests is unified in `src/storage/artifact-storage-context.js`; the format contract still comes from the manifest.
- Synthetic direct-source manifests are constructed through `src/tooling/source-artifact-adapter.js` and `src/tooling/source-runtime-bundle.js`. Supported raw-source inputs today are `safetensors` and `gguf`; `.tflite` is not implemented and must fail closed.

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
- Manifests must include explicit `session` and `execution` (v1 execution graph):
  - `session.compute.defaults.{activationDtype,mathDtype,accumDtype,outputDtype}`
  - `session.compute.kernelProfiles`
  - `session.kvcache` (nullable, but explicit)
  - `session.decodeLoop` (nullable, but explicit)
  - pinned `kernelRef` for each non-cast execution step

## Related implementation

- `src/formats/rdrr/parsing.js`
- `src/formats/rdrr/validation.js`
- `src/formats/rdrr/manifest.js`
- `src/storage/artifact-storage-context.js`
- `src/storage/shard-manager.js`
- `src/storage/downloader.js`

## Kernel override note

For override and compatibility policy, use the canonical section in [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility).
