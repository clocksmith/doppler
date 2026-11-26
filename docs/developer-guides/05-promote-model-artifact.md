# Promote a Verified Runtime Artifact

## Goal

Move a verified runtime artifact into catalog metadata, external-volume storage, and hosted publication workflows.

## When To Use This Guide

- The artifact already converts, loads, and produces coherent output.
- You are ready to treat the result as reusable and externally visible.

## Blast Radius

- Metadata + publication workflow

## Required Touch Points

- External-volume artifact directory (`$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>/`)
- `models/catalog.json` (repo model registry metadata — no weights)
- `docs/model-support-matrix.md` via sync tooling
- Artifact identity inventory via sync/check tooling
- Manifest `artifactIdentity` and catalog identity fields:
  `sourceCheckpointId`, `weightCapsuleId`, `manifestVariantId`,
  `artifactCompleteness`, `runtimePromotionState`, `weightsRefAllowed`

## Checklist

The gate and the step order are canonical in the
[Model Promotion Playbook](../model-promotion-playbook.md#promotion-workflow);
registry tool mechanics are in [registry-workflow.md](../registry-workflow.md).
Before starting, confirm:

- [ ] local convert, verify, and debug work is finished and a human reviewed the deterministic output
- [ ] raw SafeTensors/GGUF release candidates have a persisted direct-source manifest (`node tools/materialize-source-manifest.js <source-path>`)
- [ ] manifest-only updates declare a valid `weightsRef` whose weight capsule is already hosted and checker-verified; otherwise every manifest-declared file is published

## Verification

- `npm run ci:catalog:check`
- `npm run support:matrix:sync` if catalog changed
- `npm run support:inventory:sync` if catalog, compare profiles, claim lanes,
  release claims, or benchmark policy changed
- `npm run artifact-identity:check` if external-volume tracking changed
- Re-run a verify or debug pass against the repo-local or hosted artifact if publication changed the delivery path

## Common Misses

- Updating `models/catalog.json` before the output has been coherence-reviewed by a human.
- Editing catalog.json without verifying the external-volume artifact matches.
- Publishing a different artifact than the one that was actually tested.
- Publishing a manifest-only update without a valid hosted `weightsRef`.
- Publishing a direct-source manifest with absolute source paths instead of artifact-relative paths.
- Marking `runtimePromotionState="manifest-owned"` before the promoted behavior is actually encoded in the manifest.
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
