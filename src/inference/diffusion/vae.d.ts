import type { DiffusionModelConfig, DiffusionRuntimeConfig } from './types.js';
import type { DiffusionWeightEntry } from './weights.js';

export interface DecodeLatentsOptions {
  width: number;
  height: number;
  latentWidth: number;
  latentHeight: number;
  latentChannels: number;
  latentScale: number;
  weights?: DiffusionWeightEntry | null;
  modelConfig?: DiffusionModelConfig | null;
  runtime?: DiffusionRuntimeConfig | null;
}

export declare function decodeLatents(
  latents: Float32Array,
  options: DecodeLatentsOptions
): Promise<Uint8ClampedArray>;
