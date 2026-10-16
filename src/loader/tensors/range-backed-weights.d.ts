import type { CpuWeightBuffer, CpuTensorRangeSource } from '../../gpu/weight-buffer.js';
import type { TensorLocation } from '../loader-types.js';
import type { PerLayerInputLoaderContext } from '../per-layer-input-loader.js';

export declare function createRangeBackedWeightBuffer(
  ctx: Pick<PerLayerInputLoaderContext, 'loadShardRange' | 'resolveWeightLayout'>,
  name: string,
  location: TensorLocation
): CpuWeightBuffer | null;

export type RangeBackedTensorSourceFactory = (
  ctx: Pick<PerLayerInputLoaderContext, 'loadShardRange' | 'resolveWeightLayout'>,
  name: string,
  location: TensorLocation
) => CpuTensorRangeSource | null;
