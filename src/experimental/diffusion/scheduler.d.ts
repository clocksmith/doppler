/**
 * Diffusion scheduler.
 *
 * @module inference/pipelines/diffusion/scheduler
 */

import type { DiffusionSchedulerConfig } from './types.js';

export interface DiffusionScheduler {
  type: string;
  steps: number;
  sigmas: Float32Array | null;
  timesteps: Float32Array;
  predictionType?: string;
  sigmaData?: number;
}

export interface DiffusionSchedulerStepResult {
  prevSample: Float32Array;
  predOriginalSample: Float32Array;
}

export declare function buildScheduler(
  config: DiffusionSchedulerConfig,
  stepsOverride?: number | null
): DiffusionScheduler;

export declare function stepScmScheduler(
  config: DiffusionScheduler,
  modelOutput: Float32Array,
  timestep: number,
  sample: Float32Array,
  stepIndex?: number,
  noise?: Float32Array | null
): DiffusionSchedulerStepResult;
