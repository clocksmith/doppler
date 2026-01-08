/**
 * Tuner Config Schema
 *
 * Configuration for the kernel auto-tuner, which benchmarks different
 * workgroup sizes to find optimal configurations for each kernel type.
 * Results are cached in localStorage for persistence across sessions.
 *
 * @module config/schema/tuner
 */

/** Default tuner configuration */
export const DEFAULT_TUNER_CONFIG = {
  cacheKeyPrefix: 'doppler_kernel_tune_',
  defaultWarmupIterations: 3,
  defaultTimedIterations: 10,
};
