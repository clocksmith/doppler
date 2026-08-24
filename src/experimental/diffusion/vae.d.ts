import type { DiffusionModelConfig, DiffusionRuntimeConfig } from '../../inference/pipelines/diffusion/types.js';
import type { DiffusionWeightEntry } from '../../inference/pipelines/diffusion/weights.js';

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
  profile?: { totalMs?: number | null; timings?: Record<string, number> | null } | boolean | null;
}

export declare function decodeLatents(
  latents: Float32Array,
  options: DecodeLatentsOptions
): Promise<Uint8ClampedArray>;
