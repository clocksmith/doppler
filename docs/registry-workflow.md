# Registry Workflow

This page covers Hugging Face registry validation, publication, and derived catalog checks.

For the end-to-end promotion workflow across repo metadata, external-volume RDRR storage, and Hugging Face hosting, use [model-promotion-playbook.md](model-promotion-playbook.md).

## Architecture

Two files, one direction:

1. **External volume** (`$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/`) — source of truth for RDRR artifacts (manifests, shards, origin metadata)
2. **`models/catalog.json`** (repo) — source of truth for editorial + lifecycle metadata (labels, aliases, HF revisions, verification status)
3. **HF `Clocksmith/rdrr`** — published subset of catalog, filtered to approved entries

```
External volume scan → VOLUME_INDEX.json (what's physically on disk)

models/catalog.json (hand-edited, versioned in repo)
  ↓ cross-validated against VOLUME_INDEX.json
  ↓ filtered + published to
HF registry/catalog.json
```

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

Validation guarantees:
- every approved hosted entry has `hf.repoId`, `hf.revision`, and `hf.path`
- every approved hosted entry carries artifact identity metadata:
  `sourceCheckpointId`, `weightPackId`, `manifestVariantId`,
  `artifactCompleteness`, `runtimePromotionState`, and `weightsRefAllowed`
- the remote manifest resolves
- the remote manifest artifact identity matches the approved catalog entry
- every declared remote shard resolves
- the remote registry does not contain extra models outside the approved canonical hosted set
- the live demo will not surface non-fetchable remote registry entries

## Regenerate the external volume index

After adding or removing models on the external volume:

```bash
npm run external:index
```

This scans `rdrr/*/manifest.json` + `origin.json` and writes `VOLUME_INDEX.json` + `VOLUME_INDEX.md` on the volume.

## Publish a hosted model to Hugging Face

Publish a canonically approved artifact directory and rebuild the remote `Clocksmith/rdrr` registry from the approved hosted set in one workflow:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32
```

The publish workflow:
1. reads the model entry from `models/catalog.json`
2. uploads the artifact directory from the external volume (`rdrr/<model-id>`)
3. captures the artifact commit SHA
4. rebuilds `registry/catalog.json` from the approved catalog subset, pinning the just-published revision
5. verifies the published manifest resolve URL

Dry run:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32 --dry-run
```

Dry-run validates the local manifest and all manifest-declared artifact files before it prints the upload plan.
Always pass `--local-dir` and `--shard-dir` explicitly if the manifest or shard source differs from the default external volume.

## Lane variants and weights-ref siblings

A single Q4K weight pack can back multiple manifest variants — for example,
the f32-activation lane and a sibling f16-activation lane that reuses the
same shards. Two shapes are valid:

| Shape | `artifactCompleteness` | `weightsRefAllowed` | Manifest carries `weightsRef` | Publish flag |
| --- | --- | --- | --- | --- |
| Primary lane | `complete` | `false` | no | (default) |
| Manifest-only sibling | `weights-ref` | `true` | yes | `--manifest-only` |

Rules enforced across the catalog, the HF registry validator, and the
publish tool:

- Primary lanes publish shards alongside the manifest. Self-contained — a
  fresh client can fetch one URL and run.
- Manifest-only siblings publish `manifest.json` (and `origin.json` if
  present) only. Their `weightsRef` block points at a primary lane that
  must already be published in the same payload (matched by
  `weightPackId`). Without the primary, the sibling is rejected at
  validation time and at publish time.
- The shapes are exclusive: `artifactCompleteness=complete` requires
  `weightsRefAllowed=false`, and `artifactCompleteness=weights-ref`
  requires `weightsRefAllowed=true`.

Publish a manifest-only sibling explicitly with `--manifest-only`:

```bash
npm run registry:publish:hf -- \
  --model-id gemma-4-31b-it-text-q4k-ehf16-af16 \
  --manifest-only
```

The demo surface follows the same rule: a weights-ref sibling is shown
only when its primary lane is itself demo-eligible in the same catalog
view (matched by `weightPackId`). Removing the manifest-only sibling
from OPFS does not remove the shared weights — the demo copy reflects
this.

## Repair all approved hosted entries

If `Clocksmith/rdrr` drifted behind the local catalog or manifest contract, first validate the exact repair set without uploading:

```bash
npm run registry:publish:hf:all -- --local-root models/local --shard-root models/local --dry-run
```

After `hf auth login`, run the same command without `--dry-run` to republish every approved hosted artifact and rebuild `registry/catalog.json` after each successful upload:

```bash
npm run registry:publish:hf:all -- --local-root models/local --shard-root models/local
```

Use the external volume roots instead of `models/local` for release publishing when `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr` is mounted and up to date.

## After publishing

The publish tool writes the published commit SHA back to `hf.revision` in `models/catalog.json`.
After publishing, regenerate the derived support matrix:

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
