# RDRR Format

Canonical specification for Doppler runtime model artifacts.

## Scope

RDRR is the runtime contract consumed by Doppler loading and inference paths.

Related specs and contracts:
- [lora-format.md](lora-format.md) for adapter manifests
- [conversion-runtime-contract.md](conversion-runtime-contract.md) for conversion-vs-runtime ownership
- [getting-started.md](getting-started.md) for first-run convert/verify flow
- [distribution/rdrr-p2p-plan.md](distribution/rdrr-p2p-plan.md) for additive distributed execution planning
- [distribution/collective-transport-contract.md](distribution/collective-transport-contract.md) for collective transport rules

Non-RDRR note:
- Materialized direct-source artifacts use `manifest.json` plus raw SafeTensors/GGUF/TFLite assets and declare `metadata.sourceRuntime.mode="direct-source"`. They are not RDRR shards and use their own digest/path contract.
- Runtime artifact loading for both `rdrr` and persisted direct-source artifacts is unified in `src/storage/artifact-storage-context.js`.
- Direct-source runtime bundles are constructed through `src/tooling/source-artifact-adapter.js` and `src/tooling/source-runtime-bundle.js` as runtime-model contracts. `bundle.manifest` remains a compatibility alias for older callers, but `bundle.model` is the source of truth.
- Supported raw-source inputs today are `safetensors`, `gguf`, `.tflite`, `.task`, and `.litertlm`.
- Direct-source inputs are currently an experimental subsystem tier. See [subsystem-support-matrix.md](subsystem-support-matrix.md) for the public support contract.
- Promotion of `safetensors`/`gguf` into additional canonical proof lanes is
  tracked separately in [direct-source-proof-lanes.md](direct-source-proof-lanes.md).
- Direct-source manifests preserve source artifact storage facts. They do not accept converter-style quantization overrides; requantization requires a real convert step that emits a new RDRR artifact.
- `.tflite` / raw-TFLite `.task` direct-source support requires sibling `config.json` model metadata.
- Packed LiteRT-LM companion-tensor quantization still fails closed until a dedicated adapter exists.
- Float TFLite constants are read directly.
- Quantized TFLite constants currently support per-tensor affine `INT8`, `UINT8`, and `INT4` source weights, carried through the runtime model as explicit `tensor.sourceTransform` metadata and dequantized at load time to `F16`.
- Per-channel or otherwise unsupported LiteRT/TFLite quantization must still fail closed.

## Core structure

An RDRR artifact set contains:
- `manifest.json`
- `shard_*.bin`
- tokenizer assets where required

Optional additive surfaces:
- `manifest.integrityExtensions` for tensor-level block integrity metadata
- sibling `distributed.json` plans for topology-aware execution that still reference the canonical shards

## Artifact identity migration fields

The manifest may include additive identity metadata while Doppler migrates away
from `modelId` as the combined release/artifact/runtime identity.

- `artifactIdentity` identifies source checkpoint bytes, converted weight pack
  identity, and manifest/runtime-policy variant identity.
- `weightsRef` identifies a shared or external weight pack for a manifest
  variant that does not own its local shards.
- Legacy manifests may omit both fields during migration.
- When either field is present, its shape is validated by the RDRR parser.
- Runtime shard resolution does not yet consume `weightsRef`; incomplete local
  shard sets must still fail unless an explicit loader path supports the
  reference.

## Integrity extensions (Phase 0 additive contract)

`integrityExtensions` is optional and does not change canonical artifact
identity. It is a separate additive contract layer for verifiable partial reads.

Current supported shape:

```json
{
  "integrityExtensions": {
    "contractVersion": 1,
    "blockMerkle": {
      "blockSize": 1048576,
      "roots": {
        "model.embed_tokens.weight": "sha256:...",
        "model.layers.0.self_attn.q_proj.weight": "sha256:..."
      }
    }
  }
}
```

Rules:
- `integrityExtensions` is optional. Absence means the artifact predates Phase 0.
- `contractVersion` is validated explicitly and unsupported versions fail closed.
- `blockMerkle` roots are per tensor, not per shard.
- `blockMerkle` integrity is additive and must not silently rewrite
  `artifactIdentity`.
- Distributed plans that depend on integrity metadata bind it separately through
  `compatibility.integrityExtensionsHash`; they must not overload
  `artifactIdentity`.

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
