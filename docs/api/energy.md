# Energy API

## Purpose

Public energy-pipeline surface for energy-specific pipeline creation and Quintel helpers.

## Import Path

```js
import { createEnergyPipeline, computeQuintelEnergy } from '@simulatte/doppler/energy';
```

## Audience

Advanced users working with the energy pipeline family.

## Stability

Public, but advanced and specialized.

## Primary Exports

- `createEnergyPipeline(manifest, contexts?)`
- `EnergyPipeline`
- `mergeQuintelConfig(...)`
- `computeQuintelEnergy(...)`
- `runQuintelEnergyLoop(...)`

## Core Behaviors

- explicit energy-pipeline construction from manifest + contexts
- Quintel helpers exposed beside the pipeline surface

## Symbol Notes

- `createEnergyPipeline(...)`
- `EnergyPipeline`
- `mergeQuintelConfig(...)`
- `computeQuintelEnergy(...)`
- `runQuintelEnergyLoop(...)`

## Minimal Example

```js
import { createEnergyPipeline } from '@simulatte/doppler/energy';

const pipeline = await createEnergyPipeline(manifest, contexts);
```

## Advanced Example

```js
import {
  createEnergyPipeline,
  computeQuintelEnergy,
} from '@simulatte/doppler/energy';

const pipeline = await createEnergyPipeline(manifest, contexts);
const score = computeQuintelEnergy(config, activations);
console.log(score);
```

## Code Pointers

- energy export surface: [src/energy/index.js](../../src/energy/index.js)
- energy types: [src/energy/index.d.ts](../../src/energy/index.d.ts)
- energy pipeline: [src/inference/pipelines/energy/pipeline.js](../../src/inference/pipelines/energy/pipeline.js)

## Related Surfaces

- [Generated export inventory](reference/exports.md)
