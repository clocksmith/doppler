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

Treat these locations as separate roles, not interchangeable copies.
`$DOPPLER_EXTERNAL_MODELS_ROOT` is the external-volume root; Doppler tools
auto-detect `/Volumes/models`, `/Volumes/models2` or `/media/$USER/models` when
it is unset, but the shell commands below need it set.

1. Source checkpoint on external volume
   Example: `$DOPPLER_EXTERNAL_MODELS_ROOT/huggingface_cache/hub/.../snapshots/<revision>`
2. Temporary rebuild / repair directory
   Example: `/tmp/<model-id>-rebuild`
3. External-volume canonical RDRR artifact (source of truth for weights)
   Example: `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>`
4. Hugging Face hosted artifact (verified subset)
   Example: `clocksmith/rdrr`
5. Repo metadata (catalog only, no weights)
   Example: `models/catalog.json`

Rules:

- Do not publish from a scratch rebuild directory.
- Do not treat stale repo-local manifests as authoritative over a fresh verified rebuild.
- Publish to Hugging Face from the external-volume canonical RDRR directory.
- If repo metadata and external-volume artifact disagree, fix the artifact and metadata before publication.

## Canonical ID Discipline

Before converting or promoting, settle the canonical model ID. The ID must agree across four places:

1. `output.modelBaseId` in the conversion config (`src/config/conversion/<family>/<model>.json`)
2. External-volume artifact directory: `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<canonical-id>/`
3. `models/catalog.json` entry `modelId`
4. HF path: `models/<canonical-id>` under `clocksmith/rdrr`

By default the publish tool (`tools/publish-hf-registry-model.js`) takes `manifest.json` and `origin.json` from the repo-local cache `models/local/<model-id>/` and the shards and other files from `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>/`. Always pass `--local-dir $DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>` so the manifest also comes from the external-volume canonical artifact. See [Registry Workflow](registry-workflow.md#publish-a-hosted-model-to-hugging-face).

Naming schemes differ between old artifacts (e.g. `wq4k-ef16-hf16` suffix) and canonical conversion configs (e.g. `q4k-ehf16-af32` suffix). The canonical scheme is written down in [Model Naming](style/model-naming.md). Do not promote a non-canonical artifact by renaming it. Re-convert with the canonical config so the artifact and the ID agree from the start.

## Promotion Gate

This section is the canonical promotion gate. A model is promotion-ready only if
all of the following are true:

1. Source-of-truth checkpoint is identified on the external volume.
2. Conversion config is reproducible and lives under `src/config/conversion/`.
3. Manifest contract is valid and the manifest matches the config.
4. Shard bytes match manifest hashes.
5. A deterministic inference or embedding check passes on a real runtime surface.
6. The semantic boundary comparison gate passes
   ([Evidence Loop](evidence-loop.md#2-compare-semantic-boundaries)): the
   `doppler boundary compare` receipt has `promotionGate.passed: true`, which
   requires compatible boundaries, exact deterministic token parity for at
   least 128 generated tokens, and, for quantized artifacts, a passing F16 or
   source-precision control receipt (`--source-control`).
7. When the model needs cross-runtime portability (for example Doe/Cerebras
   lowering), Program Bundle export succeeds with a browser/WebGPU reference
   transcript (`npm run program-bundle:reference`, `npm run program-bundle:check`).
8. Any benchmark claim points to one saved artifact and one reproducible command.
9. The observed output has been reviewed by a human before hosted publication.

After the gate passes, follow the [Promotion Workflow](#promotion-workflow)
below. If a checked-in Program Bundle example changed, also run
`npm run program-bundle:check`.

For text models, "passes" means more than "process did not crash". The run must produce non-empty, non-collapsed, non-garbage output on a deterministic prompt.

When promotion includes a performance claim, a correctness-clean candidate that
trails its reference engine must follow the
[correct-but-behind performance trigger](https://github.com/clocksmith/doppler/blob/4b28d7831f12884c53e3a4c6e3470ca0dea9775f/docs/model-failure-action-plan.md#correct-but-behind-performance-trigger).
Near-parity claims require the paired stopping rule in
[Benchmark Methodology](benchmark-methodology.md#near-parity-statistical-protocol),
and the published score must come from a saved receipt rather than a hand-edited
scoreboard row.

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
SOURCE_DIR=$DOPPLER_EXTERNAL_MODELS_ROOT/huggingface_cache/hub/models--google--gemma-3-270m-it/snapshots/<revision>
```

### 2. Rebuild into a temporary directory

Convert or repair into `/tmp` first.

```bash
node tools/convert-safetensors-node.js "$SOURCE_DIR" \
  --config src/config/conversion/<family>/<model>.json \
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
node src/cli/doppler-cli.js debug \
  --config '{"request":{"modelId":"<model-id>","modelUrl":"file:///tmp/<model-id>-rebuild","loadMode":"http","captureOutput":true},"run":{"surface":"node"}}' \
  --runtime-config '{"shared":{"tooling":{"diagnostics":"always"}},"inference":{"prompt":"What color is the sky on a clear day? Answer in one word.","batching":{"maxTokens":4},"sampling":{"temperature":0,"topP":1,"topK":1,"repetitionPenalty":1,"greedyThreshold":0}}}' \
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

Update `models/catalog.json` to match the verified artifact. Do not copy weight shards into the repo tree — the external volume is the source of truth for weights.

Rules:

- `models/catalog.json` must point to the current HF revision after publication, not the previous one
- if a model is not correctness-clean, reflect that in human-facing status notes rather than implying full health
- `models/catalog.json` is the source of truth for editorial + lifecycle metadata
- the external volume is the source of truth for RDRR artifacts (manifests, shards, origin metadata)

### 6. Sync the external-volume canonical artifact

Copy the verified rebuild into the canonical RDRR path on the external volume.

Example:

```bash
cp -a /tmp/<model-id>-rebuild/. $DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>/
```

After copy:

- compare canonical external manifest against the repo-local manifest
- verify the external-volume directory is the exact artifact to be hosted

### 7. Verify the publish tool gate before publishing

`tools/publish-hf-registry-model.js` reads the catalog entry, then calls
`assertPromotionReady()` before uploading anything. Publication requires:

- `hf.repoId` and `hf.path`
- `sourceCheckpointId`, `weightCapsuleId`, and `manifestVariantId`
- `runtimePromotionState === "manifest-owned"`
- `artifactCompleteness === "complete"` with `weightsRefAllowed === false`, or,
  for `--manifest-only`, `artifactCompleteness === "weights-ref"` with
  `weightsRefAllowed === true`
- `lifecycle.availability.hf === true` (republish), or `false` with
  `--bootstrap` (first publish only)
- `lifecycle.status.runtime === "active"`
- `lifecycle.status.tested === "verified"`
- `lifecycle.tested.contracts.executionContractOk === true`

A brand-new model is in the bootstrap state: `hf: { repoId, path }` with no
`revision` key yet and `lifecycle.availability.hf: false`, with the other
lifecycle fields above already set.

Bootstrap, manifest-only `weightsRef` republish, and dry-run mechanics are in
[Registry Workflow](registry-workflow.md#publish-a-hosted-model-to-hugging-face).

### 7a. Publish to Hugging Face from the external-volume canonical path

Dry run first, then publish without `--dry-run`:

```bash
node tools/publish-hf-registry-model.js \
  --model-id <model-id> \
  --local-dir $DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id> \
  --dry-run
```

Passing `--local-dir` makes the uploaded manifest the one stored with the
external-volume source of truth.

### 7b. Verify the remote artifact identity after publishing

After the upload succeeds, fetch the remote manifest and confirm identity against the local artifact. Checking shard count or file sizes alone is not sufficient — verify the manifest `modelId` and at least the first shard hash against the local manifest.

If they differ, do not proceed with catalog or support-matrix updates.

### 8. Update external trackers

Each model directory on the external volume must have an `origin.json` file or
equivalent manifest provenance metadata. `npm run artifact-identity:check`
fails if provenance or artifact identity is incomplete.

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

After all `origin.json` files are present, audit the artifact identity
inventory; see
[Registry Workflow](registry-workflow.md#audit-external-volume-artifact-identity).

### 9. Re-pin repo metadata to the published HF revision

The publish tool writes the published commit SHA to `hf.revision` in
`models/catalog.json`. Confirm it, then run the follow-up sync and checks in
[Registry Workflow](registry-workflow.md#after-publishing).

## Ghost Model Discipline

A ghost model is a model ID referenced in config assets, tests, or fixtures that has no corresponding local artifact. Ghost models produce misleading failures — debug runs complete with `ok: true` but emit garbage tokens because the manifest contract validation fails silently.

Before promoting a new model ID, search for in-tree references to the old or non-canonical ID:

- `src/config/runtime/` — verify profiles
- `tests/` — regression and contract tests
- `benchmarks/vendors/fixtures/` — compare-engine fixtures

If references exist for a non-canonical or non-converted variant, either:
1. Convert that variant to make it real, or
2. Update the references to the canonical ID before promotion.

Do not leave ghost model IDs in place and route around them.

## Do Not Do This

- Do not publish with the default `models/local/<model-id>` manifest if the external-volume canonical copy differs; pass `--local-dir`.
- Do not trust a manifest whose shard hashes do not match the actual shard bytes.
- Do not call a model "working" because `ok: true` came back from `debug`.
- Do not update Hugging Face first and repair repo/external metadata later.
- Do not leave external-volume trackers stale after a promotion.

## Short Checklist

- Rebuild from external source checkpoint.
- Verify deterministic output on browser/WebGPU or the intended production surface.
- Human-review the observed output.
- Copy verified artifact to external volume.
- Update `models/catalog.json` with lifecycle metadata + HF pointers.
- Publish from the external-volume artifact.
- Confirm `hf.revision` in `models/catalog.json` is the published commit SHA.
- Run `npm run artifact-identity:check -- --root $DOPPLER_EXTERNAL_MODELS_ROOT/rdrr` to
  verify external-volume identity.
- Run `npm run ci:catalog:check` for hosted/catalog validation.
