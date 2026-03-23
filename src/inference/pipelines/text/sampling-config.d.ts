/**
 * Resolved sampling configuration with all fields defined.
 *
 * This is the single source of truth for sampling parameters after merging
 * per-call options with runtime config defaults.
 */
export interface ResolvedSamplingConfig {
  /** Sampling temperature. 0 = greedy. Must be >= 0. */
  temperature: number;

  /** Top-p (nucleus) sampling threshold. Must be in (0, 1]. */
  topP: number;

  /** Top-k sampling. Must be >= 1 (integer). */
  topK: number;

  /** Repetition penalty multiplier. 1.0 = no penalty. */
  repetitionPenalty: number;

  /** Number of recent tokens considered for repetition penalty. */
  repetitionPenaltyWindow: number;

  /** Temperature below this uses greedy decoding. */
  greedyThreshold: number;
}

/**
 * Per-call sampling options. All fields are optional; missing fields fall
 * through to runtime config defaults, then to schema defaults.
 */
export interface SamplingCallOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
}

/**
 * Merge per-call sampling options with runtime config defaults.
 * Returns a complete, validated sampling config object.
 *
 * Precedence: per-call opts > runtimeConfig.inference.sampling > schema defaults.
 *
 * Invalid values are clamped to defaults with a warning log rather than
 * throwing, so this is a non-breaking addition.
 *
 * @param opts - Per-call options (temperature, topK, topP, etc.)
 * @param runtimeConfig - Runtime config with inference.sampling defaults
 * @returns Complete sampling config with all fields defined
 */
export function resolveSamplingConfig(
  opts: SamplingCallOptions | null | undefined,
  runtimeConfig: { inference?: { sampling?: Partial<ResolvedSamplingConfig> } } | null | undefined
): ResolvedSamplingConfig;
