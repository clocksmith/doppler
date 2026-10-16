# Advanced Export Map

## Purpose

Map the older broad root surface to the current dedicated advanced subpaths.

## Current Shape

```js
import { dr } from 'doppler-gpu';
import { DopplerLoader } from 'doppler-gpu/loaders';
import { KVCache } from 'doppler-gpu/orchestration';
import { createPipeline } from 'doppler-gpu/generation';
import { getTrainingCapabilities, trainSftLoRA } from 'doppler-gpu/training';
```

## Audience

Advanced consumers migrating from the older catch-all root barrel or choosing the right dedicated subpath for new code.

## Stability

Public guidance only. The root facade is intentionally minimal now.
Subpath support tiers are defined in [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Subpath Split

Use the [generated export inventory](reference/exports.md) for exact symbol names.
The API pages explain each supported entrypoint:

| Subpath | API contract |
| --- | --- |
| `doppler-gpu/loaders` | [Loader setup](loaders.md) |
| `doppler-gpu/orchestration` | [Caches, routers, adapters and orchestration](orchestration.md) |
| `doppler-gpu/generation` | [Pipeline construction and generation](generation.md) |
| `doppler-gpu/training` | [Training and backend capabilities](training.md) |
| `doppler-gpu/tooling` | [Generated tooling exports](reference/exports.md) |
| `doppler-gpu/tooling-experimental` | [Generated experimental exports](reference/exports.md) |

## Migration Rule

- Application code should import `dr` from `doppler-gpu`.
- Explicit loader work should import from `doppler-gpu/loaders`.
- KV cache, routers, adapters, and logit-merge helpers should import from `doppler-gpu/orchestration`.
- Direct pipeline construction should import from `doppler-gpu/generation`.
- SFT/LoRA orchestration and backward primitives should import from `doppler-gpu/training`.
- Tooling and command helpers should import from `doppler-gpu/tooling`.
- Browser conversion/file-picker and P2P helpers should import from `doppler-gpu/tooling-experimental`.
- Treat `doppler-gpu/tooling` as a mixed-tier export mainly because operator command flows remain experimental.
- Check the subsystem support matrix before treating every exported subpath as equally promoted.

## Code Pointers

- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)
- loaders subpath: [src/loaders/index.js](../../src/loaders/index.js)
- orchestration subpath: [src/experimental/orchestration/index.js](../../src/experimental/orchestration/index.js)
- generation subpath: [docs/api/generation.md](generation.md)
- training subpath: [docs/api/training.md](training.md)

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
