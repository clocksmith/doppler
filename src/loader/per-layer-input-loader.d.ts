import type { WeightBuffer } from '../gpu/weight-buffer.js';
import type { ArchitectureSchema, WeightLayout } from '../config/schema/index.js';
import type { TensorLocation, PerLayerInputWeights } from './loader-types.js';

export interface PerLayerInputLoaderContext {
  modelId?: string | null;
  tensorLocations: Map<string, TensorLocation>;
  loadTensor: (
    name: string,
    toGPU?: boolean,
    silent?: boolean
  ) => Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null>;
  shouldStreamLargeWeight?: (
    name: string,
    loc: TensorLocation,
    label: string
  ) => boolean;
  resolveWeightLayout: (loc: TensorLocation) => WeightLayout;
}

export declare function loadPerLayerInputWeights(
  ctx: PerLayerInputLoaderContext,
  architecture: ArchitectureSchema | null | undefined
): Promise<PerLayerInputWeights | null>;
