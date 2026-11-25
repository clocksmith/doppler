import { canonicalizeLegacyEnumValue } from '../../../config/legacy-enum-values.js';

// Scheduler types the GPU denoising loop implements, keyed by Doppler's
// scheduler.type with the diffusers class each one mirrors. A manifest that
// resolves to any other type is rejected when the pipeline initializes, not at
// the first denoising step after the text encoders have already run.
export const SUPPORTED_DIFFUSION_SCHEDULER_TYPES = Object.freeze({
  'flowmatch-euler': 'FlowMatchEulerDiscreteScheduler',
  scm: 'SCMScheduler',
});

/**
 * @param {unknown} type
 * @param {string | null} [modelClassName]
 * @returns {'flowmatch-euler' | 'scm'}
 */
export function assertSupportedDiffusionSchedulerType(type, modelClassName = null) {
  // Runtime configs written before the kebab-case rename may say flowmatch_euler.
  const canonical = canonicalizeLegacyEnumValue(type);
  if (typeof canonical === 'string' && Object.hasOwn(SUPPORTED_DIFFUSION_SCHEDULER_TYPES, canonical)) {
    return /** @type {'flowmatch-euler' | 'scm'} */ (canonical);
  }
  const implemented = Object.entries(SUPPORTED_DIFFUSION_SCHEDULER_TYPES)
    .map(([key, className]) => `${key} (${className})`)
    .join(', ');
  const origin = modelClassName ? `model scheduler class "${modelClassName}"` : 'runtime scheduler default';
  throw new Error(
    `Diffusion scheduler type "${String(type)}" from ${origin} is not implemented by the GPU runtime. ` +
    `Implemented: ${implemented}.`
  );
}
