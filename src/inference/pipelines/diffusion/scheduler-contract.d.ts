/**
 * Scheduler types the GPU denoising loop implements.
 *
 * @module inference/pipelines/diffusion/scheduler-contract
 */

export declare const SUPPORTED_DIFFUSION_SCHEDULER_TYPES: Readonly<{
  flowmatch_euler: 'FlowMatchEulerDiscreteScheduler';
  scm: 'SCMScheduler';
}>;

/** Throws when `type` is not implemented; returns it otherwise. */
export declare function assertSupportedDiffusionSchedulerType(
  type: unknown,
  modelClassName?: string | null
): 'flowmatch_euler' | 'scm';
