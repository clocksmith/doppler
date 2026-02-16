import type { DiffusionModelConfig } from './types.js';
import type { WeightBuffer } from '../../../gpu/weight-buffer.js';

export interface SD3WeightResolver {
  get: (name: string) => GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null;
  key: (name: string) => string;
  shape: (name: string) => number[] | null;
  dtype: (name: string) => string | null;
  format: 'diffusers' | 'doppler';
}

export interface SD3WeightEntry {
  weights: Map<string, GPUBuffer | WeightBuffer | Float32Array | Uint8Array>;
  shapes: Map<string, number[]>;
  dtypes?: Map<string, string>;
}

export declare function createSD3WeightResolver(
  weightsEntry: SD3WeightEntry,
  modelConfig?: DiffusionModelConfig | null
): SD3WeightResolver;
