# Generation API

## Purpose

Lower-level text pipeline construction and utilities for consumers who need direct pipeline access instead of the root `doppler` facade.

## Import Path

```js
import { createPipeline, InferencePipeline } from 'doppler-gpu/generation';
```

## Audience

Advanced runtime consumers and experiments that need direct pipeline objects.

## Stability

Public, but advanced. Prefer the root facade unless you need direct pipeline control.
Current support tier: `tier1` advanced surface. See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- `createPipeline(manifest, contexts?)`
- `InferencePipeline`
- `EmbeddingPipeline`
- core text-pipeline option/result/types from the shipped `.d.ts`

## Core Behaviors

- explicit manifest + context driven pipeline construction
- direct access to text inference pipeline objects
- lower-level than the root `doppler` facade

## Symbol Notes

### Construction

- `createPipeline(...)`

### Pipeline classes and advanced types

- `InferencePipeline`
- `EmbeddingPipeline`
- `GenerateOptions`
- `GenerationResult`
- `PipelineContexts`
- `KVCacheSnapshot`

## Minimal Example

```js
import { createPipeline } from 'doppler-gpu/generation';

const pipeline = await createPipeline(manifest, contexts);
const text = await pipeline.generateText('Hello');
console.log(text);
```

## Advanced Example

## Code Pointers

- generation export surface: [src/generation/index.js](../../src/generation/index.js)
- generation types: [src/generation/index.d.ts](../../src/generation/index.d.ts)
- text pipeline implementation: [src/inference/pipelines/text.js](../../src/inference/pipelines/text.js)

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generated export inventory](reference/exports.md)
