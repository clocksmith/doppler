/**
 * Diffusion scheduler scaffold.
 *
 * @module inference/diffusion/scheduler
 */

export interface DiffusionScheduler {
  type: string;
  steps: number;
  sigmas: Float32Array;
}

export declare function buildScheduler(
  config: { type?: string; numSteps?: number } | null | undefined,
  stepsOverride?: number | null
): DiffusionScheduler;
