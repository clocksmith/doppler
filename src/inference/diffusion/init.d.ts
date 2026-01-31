/**
 * Diffusion init scaffold.
 *
 * @module inference/diffusion/init
 */

import type { DiffusionRuntimeConfig, DiffusionModelConfig } from './types.js';

export interface DiffusionInitState {
  modelConfig: DiffusionModelConfig;
  runtime: DiffusionRuntimeConfig;
  latentScale: number;
  latentChannels: number;
}

export declare function mergeDiffusionConfig(
  baseConfig: DiffusionRuntimeConfig | undefined | null,
  overrideConfig: DiffusionRuntimeConfig | undefined | null
): DiffusionRuntimeConfig;

export declare function initializeDiffusion(
  manifest: { config?: Record<string, unknown> },
  runtimeConfig: { inference?: { diffusion?: DiffusionRuntimeConfig } }
): DiffusionInitState;
