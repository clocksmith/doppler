/**
 * Scheduler types the GPU denoising loop implements.
 *
 * @module inference/pipelines/diffusion/scheduler-contract
 */

export declare const SUPPORTED_DIFFUSION_SCHEDULER_TYPES: Readonly<{
  'flowmatch-euler': 'FlowMatchEulerDiscreteScheduler';
  scm: 'SCMScheduler';
}>;

/** Throws when `type` is not implemented; returns its kebab-case form (flowmatch_euler is read as flowmatch-euler). */
export declare function assertSupportedDiffusionSchedulerType(
  type: unknown,
  modelClassName?: string | null
): 'flowmatch-euler' | 'scm';
