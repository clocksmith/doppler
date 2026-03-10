# Promote a Verified Runtime Artifact

## Goal

Move a locally verified runtime artifact into catalog metadata, repo-local storage, and hosted publication workflows.

## When To Use This Guide

- The artifact already converts, loads, and produces coherent output.
- You are ready to treat the result as reusable and externally visible.

## Blast Radius

- Metadata + publication workflow

## Required Touch Points

- `models/local/<model-id>/manifest.json` or a materialized direct-source manifest
- `models/catalog.json`
- External-volume artifact directory
- `docs/model-support-matrix.md` via sync tooling
- External RDRR index via sync tooling

## Recommended Order

1. Finish local convert, verify, and debug work first.
2. Get human review on deterministic output quality before touching catalog or hosted state.
3. For raw SafeTensors/GGUF release candidates, materialize the persisted direct-source manifest with `node tools/materialize-source-manifest.js <source-path>`.
4. Sync the verified manifest into `models/local/<model-id>/manifest.json`.
5. Update `models/catalog.json`, including `artifact.format` (`rdrr` or `direct-source`) and `artifact.sourceRuntimeSchema` for direct-source artifacts.
6. Run support-matrix and external-index sync if catalog or external storage changed.
7. Run catalog validation before any Hugging Face publication.
8. Publish with `npm run registry:publish:hf` only after the metadata state is clean.

## Verification

- `npm run ci:catalog:check`
- `npm run support:matrix:sync` if catalog changed
- `npm run external:rdrr:index` if external-volume tracking changed
- Re-run a verify or debug pass against the repo-local or hosted artifact if publication changed the delivery path

## Common Misses

- Updating `models/catalog.json` before the output has been coherence-reviewed by a human.
- Publishing a different artifact than the one that was actually tested.
- Publishing a direct-source manifest with absolute source paths instead of artifact-relative paths.
- Forgetting to sync derived docs and indexes after catalog changes.
- Treating load success as a publication gate without checking actual model output.

## Related Guides

- [04-conversion-config.md](04-conversion-config.md)
- [composite-model-family.md](composite-model-family.md)

## Canonical References

- [../model-promotion-playbook.md](../model-promotion-playbook.md)
- [../registry-workflow.md](../registry-workflow.md)
- `models/catalog.json`
- `package.json`
