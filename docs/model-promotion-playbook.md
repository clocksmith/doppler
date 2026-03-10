# Model Promotion Playbook

Canonical workflow for taking a model that actually works in Doppler and syncing it to:

- repo-visible `models/` metadata
- external-volume RDRR storage
- Hugging Face hosted RDRR

Use this playbook when a model has been converted, repaired, or re-verified and needs to become a maintained artifact instead of an ad hoc local experiment.

## Scope

This playbook covers text and embedding RDRR artifacts promoted through the current Doppler workflow.

It does not replace:

- [getting-started.md](getting-started.md) for first-run conversion and verify
- [conversion-runtime-contract.md](conversion-runtime-contract.md) for conversion-static vs runtime-overridable ownership
- [registry-workflow.md](registry-workflow.md) for hosted registry validation and publication mechanics

## Canonical Storage Order

Treat these locations as separate roles, not interchangeable copies:

1. Source checkpoint on external volume
   Example: `/media/x/models/huggingface_cache/hub/.../snapshots/<revision>`
2. Temporary rebuild / repair directory
   Example: `/tmp/<model-id>-rebuild`
3. Repo metadata / repo-local fallback
   Example: `models/local/<model-id>/manifest.json`
4. External-volume canonical RDRR artifact
   Example: `/media/x/models/rdrr/<model-id>`
5. Hugging Face hosted artifact
   Example: `Clocksmith/rdrr`

Rules:

- Do not publish from a scratch rebuild directory.
- Do not treat stale repo-local manifests as authoritative over a fresh verified rebuild.
- Publish to Hugging Face from the external-volume canonical RDRR directory.
- If repo metadata and external-volume artifact disagree, fix the artifact and metadata before publication.

## Canonical ID Discipline

Before converting or promoting, settle the canonical model ID. The ID must agree across four places:

1. `output.modelBaseId` in the conversion config (`tools/configs/conversion/<family>/<model>.json`)
2. Local artifact directory: `models/local/<canonical-id>/`
3. `models/catalog.json` entry `modelId`
4. HF path: `models/<canonical-id>` under `Clocksmith/rdrr`

The publish tool (`tools/publish-hf-registry-model.js`) derives `localDir` as `models/local/{modelId}` by default. If the artifact directory name does not match the catalog `modelId`, the publish step will try to upload from a non-existent path.

Naming schemes differ between old artifacts (e.g. `wq4k-ef16-hf16` suffix) and canonical conversion configs (e.g. `q4k-ehf16-af32` suffix). Do not promote a non-canonical artifact by renaming it. Re-convert with the canonical config so the artifact and the ID agree from the start.

## Promotion Gate

A model is promotion-ready only if all of the following are true:

1. Source-of-truth checkpoint is identified on the external volume.
2. Conversion config is reproducible and lives under `tools/configs/conversion/`.
3. Manifest contract is valid.
4. Shard bytes match manifest hashes.
5. A deterministic inference or embedding check passes on a real runtime surface.
6. The observed output has been reviewed by a human before hosted publication.

For text models, "passes" means more than "process did not crash". The run must produce non-empty, non-collapsed, non-garbage output on a deterministic prompt.

## Working Definition

Use these buckets consistently:

- `working`: deterministic verification passed and output is coherent enough for intended use
- `runtime-pass`: model loads and runs, but output quality is still suspect
- `broken`: contract failure, load failure, crash, NaN/finiteness loop, or clearly unusable output
- `unverified`: artifact exists, but current Doppler runtime has not re-verified it

Do not promote `runtime-pass` as `working`.

## Promotion Workflow

### 1. Start from the external source checkpoint

Use the checkpoint stored on the external volume, not a random local copy.

Example:

```bash
SOURCE_DIR=/media/x/models/huggingface_cache/hub/models--google--gemma-3-270m-it/snapshots/<revision>
```

### 2. Rebuild into a temporary directory

Convert or repair into `/tmp` first.

```bash
node tools/convert-safetensors-node.js "$SOURCE_DIR" \
  --config tools/configs/conversion/<family>/<model>.json \
  --output-dir /tmp/<model-id>-rebuild
```

Why:

- avoids mutating canonical copies before verification
- makes byte-for-byte diffing easy
- preserves a clean failure boundary

### 3. Verify the rebuilt artifact

Minimum checks:

```bash
test -f /tmp/<model-id>-rebuild/manifest.json
```

```bash
node tools/doppler-cli.js debug \
  --config '{"request":{"modelId":"<model-id>","modelUrl":"file:///tmp/<model-id>-rebuild","loadMode":"http","captureOutput":true},"run":{"surface":"node"}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"investigate"}},"inference":{"prompt":"What color is the sky on a clear day? Answer in one word.","batching":{"maxTokens":4},"sampling":{"temperature":0,"topP":1,"topK":1,"repetitionPenalty":1,"greedyThreshold":0}}}' \
  --json
```

Use the Node surface for this local rebuild verification path. Browser relay
does not share the same local-filesystem contract for `file://` artifact roots.

Required review points:

- `result.output`
- `result.metrics.tokensGenerated`
- `result.reportInfo.path`
- manifest hash fields vs actual shard bytes

If the artifact only "loads" but output is incoherent, stop here.

### 4. Human review before promotion

Before publishing or updating catalog-visible metadata, capture:

- exact deterministic prompt used
- exact observed output
- whether the result is `working`, `runtime-pass`, or `broken`

Do not silently upgrade a model to hosted/demo-visible status based only on successful execution.

### 5. Sync repo metadata

Update repo-visible model metadata to match the verified artifact:

- `models/local/<model-id>/manifest.json`
- `models/catalog.json`

Rules:

- repo-local manifest must match the actual promoted artifact
- `models/catalog.json` must point to the current HF revision after publication, not the previous one
- if a model is not correctness-clean, reflect that in human-facing status notes rather than implying full health

### 6. Sync the external-volume canonical artifact

Copy the verified rebuild into the canonical RDRR path on the external volume.

Example:

```bash
cp -a /tmp/<model-id>-rebuild/. /media/x/models/rdrr/<model-id>/
```

After copy:

- compare canonical external manifest against the repo-local manifest
- verify the external-volume directory is the exact artifact to be hosted

### 7. Verify the publish tool gate before publishing

`tools/publish-hf-registry-model.js` calls `assertPromotionReady()` which checks exactly three catalog fields before uploading anything:

- `lifecycle.status.tested === "verified"`
- `lifecycle.tested.contracts.executionContractOk === true`
- `lifecycle.tested.contracts.executionV0GraphOk === true`

The tool also requires `hf.repoId` and `hf.path` in the catalog entry. `hf.revision` is not required before publish — it is written to the remote registry by the publish step and must be recorded in the local catalog afterward (step 9).

Run a dry run to confirm the upload plan before uploading:

```bash
node tools/publish-hf-registry-model.js --model-id <model-id> --dry-run
```

### 7a. Publish to Hugging Face from the external-volume canonical path

Dry run first:

```bash
node tools/publish-hf-registry-model.js \
  --model-id <model-id> \
  --local-dir /media/x/models/rdrr/<model-id> \
  --dry-run
```

Then publish:

```bash
node tools/publish-hf-registry-model.js \
  --model-id <model-id> \
  --local-dir /media/x/models/rdrr/<model-id>
```

This guarantees the uploaded artifact is the same one stored as the external-volume source of truth.

### 7b. Verify the remote artifact identity after publishing

After the upload succeeds, fetch the remote manifest and confirm identity against the local artifact. Checking shard count or file sizes alone is not sufficient — verify the manifest `modelId` and at least the first shard hash against the local manifest.

If they differ, do not proceed with catalog or support-matrix updates.

### 8. Update external trackers

Each model directory on the external volume must have an `origin.json` file. `tools/sync-external-rdrr-index.js` throws if `origin.json` (or equivalent `manifest.metadata.sourceModel`/`variant`) is missing — the index cannot be regenerated until all models have provenance metadata.

Required fields in `origin.json`:

```json
{
  "sourceModel": "<base-model-name>",
  "sourceRepo": "<hf-org>/<repo>",
  "sourceFormat": "safetensors",
  "sourceRevision": "<snapshot-sha>",
  "variant": "<artifact-variant-suffix>",
  "convertedAt": "<iso-timestamp>"
}
```

After all `origin.json` files are present, regenerate the external RDRR tracker files:

```bash
node tools/sync-external-rdrr-index.js
```

This updates:

- `/media/x/models/RDRR_INDEX.json`
- `/media/x/models/RDRR_INDEX.md`

If there are other external-volume status docs, update them in the same promotion change.

### 9. Re-pin repo metadata to the published HF revision

After publication, update:

- `models/catalog.json`

with the new hosted revision returned by the publish step.

Then run the catalog validation flow:

```bash
npm run ci:catalog:check
```

If catalog-derived docs changed:

```bash
npm run support:matrix:sync
```

## Ghost Model Discipline

A ghost model is a model ID referenced in presets, tests, or fixtures that has no corresponding local artifact. Ghost models produce misleading failures — debug runs complete with `ok: true` but emit garbage tokens because the manifest contract validation fails silently.

Before promoting a new model ID, search for in-tree references to the old or non-canonical ID:

- `src/config/presets/runtime/` — verify presets
- `tests/` — regression and contract tests
- `benchmarks/vendors/fixtures/` — compare-engine fixtures

If references exist for a non-canonical or non-converted variant, either:
1. Convert that variant to make it real, or
2. Update the references to the canonical ID before promotion.

Do not leave ghost model IDs in place and route around them.

## Do Not Do This

- Do not publish from `models/local/<model-id>` if the external-volume canonical copy differs.
- Do not trust a manifest whose shard hashes do not match the actual shard bytes.
- Do not call a model "working" because `ok: true` came back from `debug`.
- Do not update Hugging Face first and repair repo/external metadata later.
- Do not leave external-volume trackers stale after a promotion.

## Short Checklist

- Rebuild from external source checkpoint.
- Verify deterministic output on browser/WebGPU or the intended production surface.
- Human-review the observed output.
- Sync the repo-local manifest and repo metadata.
- Sync canonical external-volume artifact.
- Publish from the external-volume artifact.
- Re-pin `models/catalog.json` to the new HF revision.
- Regenerate external-volume RDRR index.
- Run hosted/catalog validation.
