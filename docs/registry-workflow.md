# Registry Workflow

This page covers Hugging Face registry validation, publication, and derived catalog checks.

For the end-to-end promotion workflow across repo metadata, external-volume RDRR storage, and Hugging Face hosting, use [model-promotion-playbook.md](model-promotion-playbook.md).

## Scope

Use this workflow when you:
- publish a new hosted RDRR artifact
- update the external support registry
- sync the repo mirror in `models/catalog.json`
- update demo-visible hosted models
- want to catch catalog/support-matrix drift before release

## Prerequisites

- repo dependencies installed
- Hugging Face CLI installed and authenticated for publish flows
- external support registry prepared with canonical metadata
- repo `models/catalog.json` synced from the external support registry when repo-local checks need a mirror

Canonical support metadata lives on the external models volume:

- `/media/x/models/DOPPLER_SUPPORT_REGISTRY.json`
- `/media/x/models/DOPPLER_SUPPORT_REGISTRY.md`

Repo-local `models/catalog.json` is the mirrored copy consumed by current repo checks and quickstart generation.

## Validate catalog and hosted registry

Run the full catalog check before release, before demo deploy, and after any hosted catalog change:

```bash
npm run ci:catalog:check
```

This runs:
- `npm run registry:sync:scripts:check`
- `npm run support:matrix:check`
- `npm run registry:hf:check`

To run only the hosted registry validation:

```bash
npm run registry:hf:check
```

`tools/check-hf-registry.js` now prefers the external canonical support registry when it exists and falls back to the repo mirror only when the external file is absent. Remote registry membership is checked against the approved hosted subset only:

- `lifecycle.availability.hf === true`
- `lifecycle.status.runtime === "active"`
- `lifecycle.status.tested === "verified"`

Validation guarantees:
- every approved hosted entry has `hf.repoId`, `hf.revision`, and `hf.path`
- the remote manifest resolves
- every declared remote shard resolves
- the remote registry does not contain extra models outside the approved canonical hosted set
- the live demo will not surface non-fetchable remote registry entries

## Publish a hosted model to Hugging Face

Publish a canonically approved artifact directory and rebuild the remote `Clocksmith/rdrr` registry from the approved hosted set in one workflow:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32
```

The publish workflow:
1. reads the model entry from the canonical external support registry when present, otherwise from the repo mirror
2. uploads the artifact directory selected by the publish plan
3. captures the artifact commit SHA
4. rebuilds `registry/catalog.json` from the approved canonical hosted subset, pinning the just-published revision for the selected model
5. verifies the published manifest resolve URL

Dry run:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32 --dry-run
```

Preferred publication source:

- by default the publisher uploads from the canonical external-volume artifact path recorded in `external.pathRelativeToVolume`
- if no canonical external artifact path is recorded, it falls back to `models/local/<modelId>`
- when the canonical artifact lives on the external volume, pass `--local-dir <external-artifact-dir>` explicitly
- publish from the canonical external-volume RDRR directory, not a scratch rebuild directory
- ensure the external-volume artifact matches the curated manifest before publication
- update external-volume trackers after publication

## After publishing

If the repo mirror changed, regenerate the derived support matrix:

```bash
npm run support:matrix:sync
```

Then rerun:

```bash
npm run ci:catalog:check
```

## Demo surface rule

The live demo should only surface models that are actually fetchable from their published remote location.

Do not rely on implicit local or curated path fallback for hosted entries. If an entry is intended for the demo:
- the remote registry entry must resolve to a real hosted base URL
- the manifest must exist
- every shard referenced by the manifest must exist
