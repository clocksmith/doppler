# Artifact Identity Migration Plan

## Goal

Migrate Doppler from `modelId` as a combined artifact/runtime/release identity
to an explicit identity stack:

```text
source checkpoint -> weight pack -> manifest variant -> release/catalog entry
```

This plan is intentionally resumable. Each phase has checkboxes, exit criteria,
and notes about what to inspect before continuing.

## Current Status

- [x] Phase 0 started with local/external artifact inventory tooling.
- [x] Additive metadata fields are defined in the manifest schema/type surface.
- [x] Inventory tooling fails manifest-only artifacts without `weightsRef`.
- [x] Demo/quickstart/catalog publication gates require explicit artifact identity.
- [x] HF publication rejects incomplete shard sets and accepts manifest-only publication only with valid hosted `weightsRef`.
- [x] Runtime promotion state is explicit in catalog metadata and release gates require `manifest-owned`.

## Problem Statement

`modelId` currently carries too many meanings:

- Release name shown in catalog/demo.
- Converted artifact directory name.
- Shard and tensor layout identity.
- Runtime execution/session policy identity.
- Source checkpoint lineage hint.

This makes two distinct cases hard to model safely.

Case A: same source checkpoint, different converted weight packs.

Example:

- `gemma-4-e2b-it-q4k-ehf16-af32`
- `gemma-4-e2b-it-q4k-ehf16-af32-int4ple`

These share `google/gemma-4-E2B-it` as source, but they do not share Doppler
shards. INT4 PLE changes `embed_tokens_per_layer.weight` materialization and
therefore requires its own weight pack.

Case B: same converted weight pack, different manifest/runtime plans.

Example:

- A stable f32 activation manifest.
- A promoted selective-f16 decode manifest that uses the same shards.

These should not duplicate shards. They should become manifest variants over a
shared weight pack with an explicit `weightsRef`.

## Target Contract

Every promoted artifact should resolve through these identities.

```json
{
  "sourceCheckpoint": {
    "id": "google/gemma-4-E2B-it@b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    "repo": "google/gemma-4-E2B-it",
    "revision": "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    "sourceFormat": "safetensors"
  },
  "weightPack": {
    "id": "gemma4-e2b-text-q4k-ehf16-int4ple-v1",
    "sourceCheckpointId": "google/gemma-4-E2B-it@b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    "modalitySet": ["text"],
    "quantization": {
      "weights": "q4k",
      "embeddings": "f16",
      "lmHead": "f16",
      "perLayerEmbeddings": "int4_per_row"
    },
    "materialization": {
      "perLayerInputs": "range_backed"
    },
    "shardSetHash": "sha256:..."
  },
  "manifestVariant": {
    "id": "gemma4-e2b-text-q4k-int4ple-af32-exec-v1",
    "weightPackId": "gemma4-e2b-text-q4k-ehf16-int4ple-v1",
    "executionGraphHash": "sha256:...",
    "sessionHash": "sha256:...",
    "stability": "experimental"
  },
  "release": {
    "modelId": "gemma-4-e2b-it-q4k-ehf16-af32-int4ple",
    "quickstart": false
  }
}
```

## Required Semantics

- [x] `sourceCheckpoint` identifies upstream source bytes and revision.
- [x] `weightPack` identifies converted Doppler tensors and shards.
- [x] `manifestVariant` identifies inference/session/execution policy for a weight pack.
- [x] `release.modelId` remains the user-facing catalog/demo id.
- [x] `weightsRef` is allowed only when a manifest variant uses an external/shared weight pack.
- [x] A manifest with inline `shards[]` must have every shard present in the same artifact root before publication/demo source selection.
- [x] A manifest without complete local shards must have a resolvable `weightsRef` on explicit runtime source paths.
- [x] Runtime profiles remain investigation/calibration overlays until explicitly promoted.
- [x] Capability transforms remain rule-driven runtime remaps until explicitly promoted.

## Proposed Manifest Additions

Add a new manifest section. Exact names can be adjusted during schema work, but
the distinction must remain.

```json
{
  "artifactIdentity": {
    "sourceCheckpointId": "google/gemma-4-E2B-it@b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    "sourceRepo": "google/gemma-4-E2B-it",
    "sourceRevision": "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    "sourceFormat": "safetensors",
    "conversionConfigPath": "src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json",
    "conversionConfigDigest": "sha256:...",
    "weightPackId": "gemma4-e2b-text-q4k-ehf16-int4ple-v1",
    "weightPackHash": "sha256:...",
    "manifestVariantId": "gemma4-e2b-text-q4k-int4ple-af32-exec-v1",
    "modalitySet": ["text"],
    "materializationProfile": "range_backed-int4ple",
    "artifactCompleteness": "complete"
  }
}
```

Add `weightsRef` only for manifest variants that share another artifact's shards.

```json
{
  "weightsRef": {
    "weightPackId": "gemma4-e2b-text-q4k-ehf16-int4ple-v1",
    "artifactRoot": "models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple",
    "manifestDigest": "sha256:...",
    "shardSetHash": "sha256:..."
  }
}
```

## Proposed Catalog Additions

Backfill catalog entries with enough identity to avoid guessing from names.

```json
{
  "modelId": "gemma-4-e2b-it-q4k-ehf16-af32-int4ple",
  "sourceCheckpointId": "google/gemma-4-E2B-it@b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
  "weightPackId": "gemma4-e2b-text-q4k-ehf16-int4ple-v1",
  "manifestVariantId": "gemma4-e2b-text-q4k-int4ple-af32-exec-v1",
  "artifactCompleteness": "complete",
  "runtimePromotionState": "manifest-owned",
  "weightsRefAllowed": false
}
```

## Phase 0: Freeze and Inventory

Objective: find every current drift case before changing runtime behavior.

- [ ] Inventory every `models/local/**/manifest.json`.
- [ ] Inventory every external RDRR root artifact if present.
- [ ] Inventory every HF `Clocksmith/rdrr` model folder.
- [ ] For each manifest with `shards[]`, check all referenced files exist.
- [ ] For each manifest without local shards, classify as invalid unless it has `weightsRef`.
- [ ] Detect sidecar manifests such as `manifest-no-ple.json`.
- [ ] Detect duplicate source checkpoints with different shard layouts.
- [ ] Detect duplicate shard sets with different manifests.
- [ ] Detect catalog entries whose model folder is metadata-only.
- [ ] Detect quickstart/demo entries that point at incomplete artifacts.

Suggested inventory output:

```json
{
  "artifactRoot": "models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple",
  "modelId": "gemma-4-e2b-it-q4k-ehf16-af32-int4ple",
  "hasManifest": true,
  "declaredShardCount": 60,
  "presentShardCount": 0,
  "hasWeightsRef": false,
  "artifactCompleteness": "incomplete",
  "classification": "invalid-manifest-only"
}
```

Exit criteria:

- [x] Inventory command exists with `--check`.
- [x] Inventory command emits JSON.
- [x] Current known incomplete artifacts are listed.
- [x] No runtime behavior has changed yet.

### Phase 0 Inventory Receipt: 2026-04-20

Command:

```bash
npm run artifact-identity:inventory -- --json --pretty
```

Summary:

- `models/local` was scanned.
- `/Volumes/models/rdrr` and `/media/x/models/rdrr` were absent on this machine.
- 11 manifest artifacts were found.
- 10 primary manifests and 1 sidecar manifest were found.
- 4 artifacts were complete.
- 6 artifacts were incomplete.
- 0 artifacts had `weightsRef`.
- 10 catalog entries were checked.
- 12 blocking errors and 11 warnings were reported.

Complete local artifacts:

- `models/local/gemma-3-1b-it-q4k-ehf16-af32`
- `models/local/gemma-3-270m-it-q4k-ehf16-af32`
- `models/local/gemma-4-e2b-it-q4k-ehf16-af32`
- `models/local/google-embeddinggemma-300m-q4k-ehf16-af32`

Incomplete local artifacts:

- `models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple`: `invalid-manifest-only`, 60 declared shards, 0 present, no `weightsRef`.
- `models/local/lfm2-5-1-2b-instruct-q4k-ehf16-af32`: `invalid-manifest-only`, 13 declared shards, 0 present, no `weightsRef`.
- `models/local/qwen-3-5-0-8b-q4k-ehaf16`: `invalid-incomplete-shards`, 18 declared shards, 15 present, missing `shard_00015.bin` through `shard_00017.bin`.
- `models/local/qwen-3-5-2b-q4k-ehaf16`: `invalid-incomplete-shards`, 32 declared shards, 27 present, missing `shard_00027.bin` through `shard_00031.bin`.
- `models/local/sana-sprint-0-6b-wf16-ef16-hf16-f16`: `invalid-manifest-only`, 105 declared shards, 0 present, no `weightsRef`.
- `models/local/translategemma-4b-it-q4k-ehf16-af32`: `invalid-manifest-only`, 47 declared shards, 0 present, no `weightsRef`.

Sidecar manifest:

- `models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest-no-ple.json`

Duplicate shard-layout group:

- `manifest.json` and `manifest-no-ple.json` under `models/local/gemma-4-e2b-it-q4k-ehf16-af32` share the same 113-shard layout.

## Phase 1: Additive Schema and Parser Fields

Objective: add metadata without requiring it immediately.

- [x] Add `artifactIdentity` to the manifest schema.
- [x] Add optional `weightsRef` to the manifest schema.
- [x] Add parser support for both fields.
- [x] Preserve current manifests as valid during this phase.
- [x] Add tests for parsing manifests with `artifactIdentity`.
- [x] Add tests for parsing manifests with `weightsRef`.
- [x] Add tests proving old manifests still parse.
- [x] Document that missing fields are legacy-compatible only during migration.

Touch points:

- `src/config/schema/`
- `src/loader/`
- `src/types/`
- `docs/rdrr-format.md`
- `docs/conversion-runtime-contract.md`

Exit criteria:

- [x] New fields round-trip through manifest parsing.
- [x] Existing artifacts still load under legacy compatibility.
- [x] No artifact is silently reclassified at runtime.

## Phase 2: Conversion Metadata Emission

Objective: make new conversions emit identity metadata.

- [x] Add source checkpoint metadata to converter output.
- [x] Add conversion config path and digest to converter output.
- [x] Derive deterministic `weightPackId`.
- [x] Derive deterministic `manifestVariantId`.
- [x] Derive deterministic `shardSetHash`.
- [x] Include modality set in emitted metadata.
- [x] Include materialization profile in emitted metadata.
- [x] Keep `modelId` as release/catalog id, not weight identity.

Suggested `weightPackId` inputs:

- Source checkpoint id.
- Source revision.
- Model family.
- Modality set.
- Weight quantization.
- Embedding quantization.
- LM head quantization.
- PLE materialization policy.
- Sharding policy.

Suggested `manifestVariantId` inputs:

- Weight pack id.
- Inference schema id.
- Execution graph hash.
- Session hash.
- Chat template id.
- Runtime-visible model parameters.

Exit criteria:

- [x] Reconverted artifacts include `artifactIdentity`.
- [x] Reconverted same config produces the same ids.
- [x] Changing only runtime execution/session changes `manifestVariantId`, not `weightPackId`.
- [x] Changing quantization/materialization changes `weightPackId`.

## Phase 3: Artifact Contract Checker

Objective: fail broken artifacts before runtime tries to fetch a shard.

- [ ] Add an artifact contract checker command or tool.
- [ ] Checker validates manifest presence.
- [ ] Checker validates all local `shards[]`.
- [ ] Checker validates shard hashes/sizes when metadata exists.
- [x] Checker validates `weightsRef` target existence.
- [x] Checker validates `weightsRef.weightPackId` matches target.
- [x] Checker validates `artifactCompleteness`.
- [x] Checker validates catalog `weightPackId` and `manifestVariantId`.
- [x] Checker validates HF publish candidates.
- [ ] Checker supports `--check` for CI.

Required failure:

```text
ArtifactContractError:
modelId gemma-4-e2b-it-q4k-ehf16-af32-int4ple declares 60 local shards,
but 60 are missing. This artifact is incomplete and has no weightsRef.
```

Exit criteria:

- [ ] Metadata-only artifact folders fail checker.
- [ ] Complete local artifacts pass checker.
- [x] Manifest variants with valid `weightsRef` pass checker.
- [x] Manifest variants with broken `weightsRef` fail checker.

## Phase 4: Loader Fail-Fast Boundary

Objective: move load failures from mid-shard fetch to artifact contract resolution.

- [x] Resolve artifact identity before model load.
- [x] If manifest has local `shards[]`, check presence before GPU allocation.
- [x] If manifest has `weightsRef`, resolve the referenced weight pack before GPU allocation on explicit runtime source paths.
- [x] Reject unresolved `weightsRef`.
- [x] Reject mismatched `weightPackId`.
- [x] Reject incomplete local artifacts.
- [x] Keep explicit `modelUrl` fail-closed.
- [x] Do not fallback to HF or another model implicitly.

Exit criteria:

- [x] Missing `shard_00009.bin` style failures report artifact contract errors.
- [x] No GPU weight buffers are allocated before artifact contract passes.
- [x] Browser and Node explicit-source surfaces report equivalent errors.

## Phase 5: Catalog and Demo Resolver Migration

Objective: make user-facing selection resolve through the identity stack.

New resolution path:

```text
catalog entry -> manifestVariantId -> weightPackId -> artifact source
```

- [x] Add identity fields to `models/catalog.json`.
- [x] Add identity fields to demo registry generation.
- [x] Demo model cards filter by complete artifact or valid hosted artifact.
- [x] Demo no longer treats every model folder as complete.
- [x] Quickstart entries require `artifactCompleteness="complete"`.
- [ ] Non-quickstart entries can be listed only if explicitly allowed and resolvable.
- [x] Local source candidates must pass artifact contract before selection.
- [x] Hosted source candidates must pass manifest/shard listing checks or declare `weightsRef`.

Exit criteria:

- [x] Demo cannot surface metadata-only INT4 PLE folders as loadable.
- [ ] Demo can surface promoted manifest variants sharing a weight pack.
- [x] Catalog remains backward-compatible for legacy entries during migration.

## Phase 6: HF Publish Gate

Objective: prevent hosted incomplete artifacts.

- [x] Publish tooling rejects manifest-only folders without `weightsRef`.
- [x] Publish tooling verifies all declared shards exist before upload.
- [x] Publish tooling verifies uploaded tree after upload.
- [x] Publish tooling writes artifact identity metadata through converter-emitted manifests and catalog gates.
- [x] Publish tooling records HF revision in catalog metadata.
- [x] Publish tooling supports manifest-only variants only with valid `weightsRef`.

Required rejection:

```text
PublishArtifactError:
models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple has manifest.json but no shards.
Add a valid weightsRef or publish the matching weight pack.
```

Exit criteria:

- [x] HF cannot contain a release-visible manifest-only artifact accidentally.
- [x] Every hosted catalog entry resolves to a complete weight pack.

## Phase 7: Backfill Existing Artifacts

Objective: classify current artifacts into the new regime.

Known examples to classify:

- [x] `gemma-4-e2b-it-q4k-ehf16-af32`
- [x] `gemma-4-e2b-it-q4k-ehf16-af32-int4ple`
- [ ] `gemma-4-e2b-it-q4k-ehf16-af32-no-ple`
- [ ] `gemma-4-e2b-it-text-q4k-ehf16-af32-rebuild-v4`
- [x] `qwen-3-5-0-8b-q4k-ehaf16`
- [x] `qwen-3-5-2b-q4k-ehaf16`
- [x] Gemma 3 quickstart artifacts.
- [x] EmbeddingGemma artifacts.
- [x] TranslateGemma artifacts.
- [x] LFM entries.

Classification rules:

- [x] Same source checkpoint plus different converted tensors means separate weight packs.
- [x] Same shards plus different execution/session means manifest variants.
- [ ] Sidecar manifest files must become named manifest variants or be removed.
- [x] `runtimeProfile`-only behavior remains unpromoted unless backed by evidence.
- [x] Local-only incomplete artifacts remain excluded from catalog/demo.

Exit criteria:

- [x] Every catalog entry has source checkpoint id.
- [x] Every catalog entry has weight pack id.
- [x] Every catalog entry has manifest variant id.
- [x] Every catalog entry has artifact completeness status.

## Phase 8: Runtime Profile Promotion Discipline

Objective: make promotion state explicit.

- [ ] Runtime profiles keep `intent`, `stability`, `owner`, and `createdAtUtc`.
- [ ] Experimental runtime profiles cannot be release identity.
- [ ] A promoted runtime profile must become a manifest variant or conversion config update.
- [ ] Benchmark/debug evidence must name the exact manifest variant.
- [ ] Catalog claims must not point at ad hoc runtime overlays.

Suggested promotion states:

- `runtime-only-investigation`
- `runtime-only-calibration`
- `candidate-manifest-variant`
- `manifest-owned`
- `conversion-owned`
- `deprecated`

Exit criteria:

- [x] Demo/release paths do not depend on experimental runtime profiles.
- [x] Promoted behavior has manifest identity.
- [x] Bench reports include manifest variant id.

## Phase 9: Enforcement and Legacy Cleanup

Objective: remove temporary compatibility once inventory is clean.

- [x] Require `artifactIdentity` for promoted artifacts.
- [x] Require `weightPackId` for promoted artifacts.
- [x] Require `manifestVariantId` for promoted artifacts.
- [x] Forbid manifest-only artifact folders without `weightsRef`.
- [x] Forbid catalog entries without identity fields.
- [x] Make artifact contract checker part of publish checks.
- [x] Make artifact contract checker part of catalog/demo generation checks.
- [ ] Remove or migrate sidecar manifests.
- [x] Document legacy aliases as release aliases, not artifact identity.

Exit criteria:

- [x] No promoted artifact relies on `modelId` as weight identity.
- [x] No incomplete artifact is release-visible.
- [x] Loader errors are contract errors, not missing-file surprises.

### Migration Receipt: 2026-04-20

Completed code/config migration:

- Converter output now emits `artifactIdentity` and optional `weightsRef`
  fields from explicit conversion inputs, source metadata, shard hashes,
  quantization/materialization policy, and manifest execution/session content.
- `models/catalog.json` is backfilled with `sourceCheckpointId`,
  `weightPackId`, `manifestVariantId`, `artifactCompleteness`,
  `runtimePromotionState`, and `weightsRefAllowed`.
- Quickstart registry generation requires complete hosted artifact identity and
  mirrors those fields into `src/client/doppler-registry.json`.
- Demo model cards require `quickstart=true`, `artifactCompleteness="complete"`,
  `runtimePromotionState="manifest-owned"`, and `weightsRefAllowed=false`
  before attempting local/HF source resolution.
- HF publish tooling accepts `--manifest-only` only for manifests with valid
  hosted `weightsRef`, verifies declared local artifact files before complete
  uploads, and requires promoted catalog identity.
- Hosted registry checks require catalog identity and verify that remote
  manifests carry matching `artifactIdentity`.

Known remaining artifact work:

- Existing hosted artifacts must be republished from reconverted manifests
  before the stricter remote registry checker will pass against old revisions.
- Runtime and hosted registry `weightsRef` loading are supported for explicit
  URL/file runtime sources and HF manifest-only publication. Shared weight-pack
  variants remain excluded from quickstart/demo OPFS promotion until
  variant-aware storage exists.
- Local incomplete artifact folders listed in the Phase 0 receipt remain
  excluded; they need reconversion or valid `weightsRef` targets.
- Sidecar manifests such as `manifest-no-ple.json` still need first-class
  manifest-variant promotion or removal.

## Legacy Issues to Track

- [ ] `models/local/**` can contain developer-local partial artifacts.
- [ ] HF `Clocksmith/rdrr` can currently contain manifest-only folders.
- [ ] `manifest-no-ple.json` style sidecars are not first-class variants.
- [ ] `rebuild-vN` names encode process history instead of contract identity.
- [ ] `int4ple` names encode materialization details not exposed structurally.
- [ ] Quickstart filtering can hide broken artifacts but does not fix identity drift.
- [ ] Runtime capability transforms can become de facto product behavior before promotion.
- [ ] Conversion config names and manifest model ids are not enough to prove shard compatibility.

## Resume Protocol

When resuming this migration:

- [ ] Read this file first.
- [ ] Read `docs/style/general-style-guide.md`.
- [ ] Read `docs/style/javascript-style-guide.md`.
- [ ] Read `docs/style/config-style-guide.md`.
- [ ] Identify the current phase.
- [ ] Run or create the inventory checker before editing individual artifact metadata.
- [ ] Do not repair one artifact manually if the same drift class appears elsewhere.
- [ ] Do not change runtime fallback behavior to mask incomplete artifacts.
- [ ] Update checkboxes in this file only after the corresponding code/config/docs change exists.

## Validation Commands

These commands are placeholders until the checker names are finalized.

```bash
npm run artifact-identity:inventory -- --json --pretty
npm run artifact-identity:check
npm run catalog:check
npm run publish:check -- --model-id MODEL_ID
```

Existing commands that should eventually include artifact identity checks:

```bash
npm run onboarding:check
npm run agents:verify
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"MODEL_ID"},"run":{"surface":"auto"}}' --json
```

## Definition of Done

- [ ] `modelId` is only release/catalog identity.
- [ ] `sourceCheckpointId` identifies upstream bytes.
- [ ] `weightPackId` identifies Doppler converted shards.
- [ ] `manifestVariantId` identifies runtime-visible manifest behavior.
- [ ] Shared-weight manifest variants use explicit `weightsRef`.
- [ ] Distinct converted layouts use distinct weight packs.
- [ ] Loader fails before shard fetch when artifact contracts are invalid.
- [ ] Demo only shows loadable artifacts or explicitly marked unavailable entries.
- [ ] HF publication cannot publish incomplete artifact folders accidentally.
- [ ] Runtime profiles are not promoted behavior unless converted into manifest variants.
