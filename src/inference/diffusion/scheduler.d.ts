/**
 * Diffusion scheduler scaffold.
 *
 * @module inference/diffusion/scheduler
 */

import type { DiffusionSchedulerConfig } from './types.js';

export interface DiffusionScheduler {
  type: string;
  steps: number;
  sigmas: Float32Array;
  timesteps: Float32Array;
}

export declare function buildScheduler(
  config: DiffusionSchedulerConfig,
  stepsOverride?: number | null
): DiffusionScheduler;
