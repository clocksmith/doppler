/**
 * Tuner Config Schema
 *
 * Configuration for the kernel auto-tuner, which benchmarks different
 * workgroup sizes to find optimal configurations for each kernel type.
 * Results are cached in localStorage for persistence across sessions.
 *
 * @module config/schema/tuner
 */

/**
 * Configuration for the kernel auto-tuner.
 *
 * Controls cache key prefixes and default iteration counts for warmup
 * and timed benchmarking passes.
 */
export interface TunerConfigSchema {
  /** Prefix for localStorage cache keys (device signature appended) */
  cacheKeyPrefix: string;

  /** Number of warmup iterations before timing (not included in measurements) */
  defaultWarmupIterations: number;

  /** Number of timed iterations to average for benchmark results */
  defaultTimedIterations: number;
}

/** Default tuner configuration */
export declare const DEFAULT_TUNER_CONFIG: TunerConfigSchema;
