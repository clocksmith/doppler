# Promote a Verified Model Artifact

## Goal

Move a locally verified RDRR artifact into curated metadata, external storage, and hosted publication workflows.

## When To Use This Guide

- The artifact already converts, loads, and produces coherent output.
- You are ready to treat the result as reusable and externally visible.

## Blast Radius

- Metadata + publication workflow

## Required Touch Points

- `models/curated/<model-id>/manifest.json`
- `models/catalog.json`
- External-volume RDRR directory
- `docs/model-support-matrix.md` via sync tooling
- External RDRR index via sync tooling

## Recommended Order

1. Finish local convert, verify, and debug work first.
2. Get human review on deterministic output quality before touching catalog or hosted state.
3. Sync the verified manifest into `models/curated/<model-id>/manifest.json`.
4. Update `models/catalog.json`.
5. Run support-matrix and external-index sync if catalog or external storage changed.
6. Run catalog validation before any Hugging Face publication.
7. Publish with `npm run registry:publish:hf` only after the metadata state is clean.

## Verification

- `npm run ci:catalog:check`
- `npm run support:matrix:sync` if catalog changed
- `npm run external:rdrr:index` if external-volume tracking changed
- Re-run a verify or debug pass against the curated or hosted artifact if publication changed the delivery path

## Common Misses

- Updating `models/catalog.json` before the output has been coherence-reviewed by a human.
- Publishing a different artifact than the one that was actually tested.
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
