# Generation API

## Purpose

Lower-level text pipeline construction and utilities for consumers who need direct pipeline access instead of the root `doppler` facade.

## Import Path

```js
import { createPipeline, InferencePipeline } from '@simulatte/doppler/generation';
```

## Audience

Advanced runtime consumers, internal integrations, and experiments that need direct pipeline objects and manifest parsing helpers.

## Stability

Public, but advanced. Prefer the root facade unless you need direct pipeline control.

## Primary Exports

- `createPipeline(manifest, contexts?)`
- `InferencePipeline`
- `EmbeddingPipeline`
- `parseModelConfig(...)`
- `parseModelConfigFromManifest(...)`
- `loadWeights(...)`
- `initTokenizer(...)`

## Core Behaviors

- explicit manifest + context driven pipeline construction
- direct access to text inference pipeline objects
- lower-level than the root `doppler` facade

## Symbol Notes

### Construction and parsing

- `createPipeline(...)`
- `parseModelConfig(...)`
- `parseModelConfigFromManifest(...)`

### Loading and tokenizer helpers

- `loadWeights(...)`
- `initTokenizer(...)`
- `initTokenizerFromManifest(...)`
- `isStopToken(...)`

### Pipeline classes and advanced types

- `InferencePipeline`
- `EmbeddingPipeline`
- structured pipeline exports re-exported from the generation surface

## Minimal Example

```js
import { createPipeline } from '@simulatte/doppler/generation';

const pipeline = await createPipeline(manifest, contexts);
const text = await pipeline.generateText('Hello');
console.log(text);
```

## Advanced Example

```js
import {
  createPipeline,
  parseModelConfigFromManifest,
  initTokenizerFromManifest,
} from '@simulatte/doppler/generation';

const parsed = parseModelConfigFromManifest(manifest);
const tokenizer = await initTokenizerFromManifest(manifest, parsed);
const pipeline = await createPipeline(manifest, { tokenizer });
```

## Code Pointers

- generation export surface: [src/generation/index.js](../../src/generation/index.js)
- generation types: [src/generation/index.d.ts](../../src/generation/index.d.ts)
- text pipeline implementation: [src/inference/pipelines/text.js](../../src/inference/pipelines/text.js)

## Related Surfaces

- [Root API](root.md)
- [Generated export inventory](reference/exports.md)
