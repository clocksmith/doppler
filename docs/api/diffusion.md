# Diffusion API

## Purpose

Public diffusion/image pipeline surface for consumers working with diffusion manifests and image regression helpers.

## Import Path

```js
import {
  createDiffusionPipeline,
  computeImageRegressionMetrics,
} from 'doppler-gpu/diffusion';
```

## Audience

Advanced users and integrations working with diffusion-specific pipeline flows.

## Stability

Public, but advanced and narrower than the text root surface.
Current support tier: `experimental`. See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- `createDiffusionPipeline(manifest, contexts?)`
- `DiffusionPipeline`
- `createDiffusionWeightLoader(...)`
- `mergeDiffusionConfig(...)`
- `initializeDiffusion(...)`
- `computeImageFingerprint(...)`
- `computeImageRegressionMetrics(...)`
- `assertImageRegressionWithinTolerance(...)`

## Core Behaviors

- explicit diffusion manifest + context construction
- weight-loader and config helpers exposed alongside pipeline creation
- image regression helpers available in the same subpath for validation workflows

## Symbol Notes

### Pipeline creation

- `createDiffusionPipeline(...)`
- `DiffusionPipeline`

### Weight and config helpers

- `createDiffusionWeightLoader(...)`
- `mergeDiffusionConfig(...)`
- `initializeDiffusion(...)`

### Regression helpers

- `computeImageFingerprint(...)`
- `computeImageRegressionMetrics(...)`
- `assertImageRegressionWithinTolerance(...)`

## Minimal Example

```js
import { createDiffusionPipeline } from 'doppler-gpu/diffusion';

const pipeline = await createDiffusionPipeline(manifest, contexts);
```

## Advanced Example

```js
import {
  createDiffusionPipeline,
  computeImageRegressionMetrics,
} from 'doppler-gpu/diffusion';

const pipeline = await createDiffusionPipeline(manifest, contexts);
const metrics = computeImageRegressionMetrics(actualImageData, expectedImageData);
console.log(metrics);
```

## Code Pointers

- diffusion export surface: [src/experimental/diffusion/index.js](../../src/experimental/diffusion/index.js)
- diffusion types: [src/experimental/diffusion/index.d.ts](../../src/experimental/diffusion/index.d.ts)
- diffusion pipeline: [src/inference/pipelines/diffusion/pipeline.js](../../src/inference/pipelines/diffusion/pipeline.js)

## Related Surfaces

- [Generated export inventory](reference/exports.md)
