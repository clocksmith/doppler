/**
 * Diffusion UNet scaffold.
 *
 * @module inference/diffusion/unet
 */

import type { DiffusionScheduler } from './scheduler.js';

export declare function runUnetStep(
  latents: Float32Array,
  scheduler: DiffusionScheduler,
  stepIndex: number,
  guidanceScale?: number
): Float32Array;
