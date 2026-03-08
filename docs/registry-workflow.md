# Registry Workflow

This page covers Hugging Face registry validation, publication, and derived catalog checks.

For the end-to-end promotion workflow across repo metadata, external-volume RDRR storage, and Hugging Face hosting, use [model-promotion-playbook.md](model-promotion-playbook.md).

## Scope

Use this workflow when you:
- publish a new hosted RDRR artifact
- update `models/catalog.json`
- update demo-visible hosted models
- want to catch catalog/support-matrix drift before release

## Prerequisites

- repo dependencies installed
- Hugging Face CLI installed and authenticated for publish flows
- local `models/catalog.json` entry prepared with canonical metadata

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
- every local `lifecycle.availability.hf=true` entry has `hf.repoId`, `hf.revision`, and `hf.path`
- the remote manifest resolves
- every declared remote shard resolves
- the live demo will not surface non-fetchable remote registry entries

## Publish a hosted model to Hugging Face

Publish a local artifact directory and patch the remote `Clocksmith/rdrr` registry in one workflow:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32
```

The publish workflow:
1. reads the canonical local entry from `models/catalog.json`
2. uploads the local artifact directory to Hugging Face
3. captures the artifact commit SHA
4. patches `registry/catalog.json` with the pinned HF revision
5. verifies the published manifest resolve URL

Dry run:

```bash
npm run registry:publish:hf -- --model-id translategemma-4b-it-q4k-ehf16-af32 --dry-run
```

Preferred publication source:

- publish from the canonical external-volume RDRR directory, not a scratch rebuild directory
- ensure the external-volume artifact matches the curated manifest before publication
- update external-volume trackers after publication

## After publishing

If the local catalog changed, regenerate the derived support matrix:

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
