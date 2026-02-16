import type { RDRRManifest } from '../../../formats/rdrr/types.js';
import type { TensorLocation } from '../../../loader/loader-types.js';
import type { WeightBuffer } from '../../../gpu/weight-buffer.js';
import type { DiffusionRuntimeConfig } from './types.js';

export interface DiffusionWeightEntry {
  weights: Map<string, GPUBuffer | WeightBuffer | Float32Array | Uint8Array>;
  shapes: Map<string, number[]>;
  dtypes: Map<string, string>;
  release: () => void;
}

export interface DiffusionWeightLoader {
  tensorLocations: Map<string, TensorLocation>;
  shardCache: unknown;
  loadTensor: (
    name: string,
    toGPU?: boolean
  ) => Promise<{
    value: GPUBuffer | WeightBuffer | Float32Array | Uint8Array;
    location: TensorLocation;
    buffers: GPUBuffer[];
  } | null>;
  loadComponentWeights: (
    componentId: string,
    options?: {
      filter?: (name: string, location: TensorLocation) => boolean;
      toGPU?: boolean;
    }
  ) => Promise<DiffusionWeightEntry>;
}

export declare function createDiffusionWeightLoader(
  manifest: RDRRManifest,
  options?: {
    baseUrl?: string | null;
    runtimeConfig?: { inference?: { diffusion?: DiffusionRuntimeConfig } };
    verifyHashes?: boolean;
  }
): Promise<DiffusionWeightLoader>;
