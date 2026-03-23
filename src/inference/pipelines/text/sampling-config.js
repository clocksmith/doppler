import { log } from '../../../debug/index.js';
import { DEFAULT_SAMPLING_DEFAULTS } from '../../../config/schema/inference-defaults.schema.js';

/**
 * Merge per-call sampling options with runtime config defaults.
 * Returns a complete, validated sampling config object.
 *
 * Precedence: per-call opts > runtimeConfig.inference.sampling > schema defaults.
 *
 * Invalid values are clamped to defaults with a warning log rather than
 * throwing, so this is a non-breaking addition.
 *
 * @param {Object} opts - Per-call options (temperature, topK, topP, etc.)
 * @param {Object} runtimeConfig - Runtime config with inference.sampling defaults
 * @returns {Object} Complete sampling config with all fields defined
 */
export function resolveSamplingConfig(opts, runtimeConfig) {
  const samplingDefaults = runtimeConfig?.inference?.sampling ?? DEFAULT_SAMPLING_DEFAULTS;

  const raw = {
    temperature: opts?.temperature ?? samplingDefaults.temperature ?? DEFAULT_SAMPLING_DEFAULTS.temperature,
    topP: opts?.topP ?? samplingDefaults.topP ?? DEFAULT_SAMPLING_DEFAULTS.topP,
    topK: opts?.topK ?? samplingDefaults.topK ?? DEFAULT_SAMPLING_DEFAULTS.topK,
    repetitionPenalty: opts?.repetitionPenalty ?? samplingDefaults.repetitionPenalty ?? DEFAULT_SAMPLING_DEFAULTS.repetitionPenalty,
    repetitionPenaltyWindow: samplingDefaults.repetitionPenaltyWindow ?? DEFAULT_SAMPLING_DEFAULTS.repetitionPenaltyWindow,
    greedyThreshold: samplingDefaults.greedyThreshold ?? DEFAULT_SAMPLING_DEFAULTS.greedyThreshold,
  };

  const resolved = {
    temperature: clampWithWarning('temperature', raw.temperature, 0, Infinity, DEFAULT_SAMPLING_DEFAULTS.temperature),
    topP: clampWithWarning('topP', raw.topP, Number.MIN_VALUE, 1, DEFAULT_SAMPLING_DEFAULTS.topP),
    topK: clampWithWarning('topK', raw.topK, 1, Infinity, DEFAULT_SAMPLING_DEFAULTS.topK),
    repetitionPenalty: raw.repetitionPenalty,
    repetitionPenaltyWindow: raw.repetitionPenaltyWindow,
    greedyThreshold: raw.greedyThreshold,
  };

  // Ensure topK is an integer
  if (Number.isFinite(resolved.topK)) {
    resolved.topK = Math.max(1, Math.floor(resolved.topK));
  }

  return resolved;
}

/**
 * Clamp a value to [min, max]. If the value is invalid, log a warning and
 * return the fallback default.
 *
 * @param {string} name - Parameter name for logging
 * @param {number} value - Raw value
 * @param {number} min - Minimum allowed value (inclusive)
 * @param {number} max - Maximum allowed value (inclusive)
 * @param {number} fallback - Default value to use if clamping fails
 * @returns {number} Clamped value
 */
function clampWithWarning(name, value, min, max, fallback) {
  if (!Number.isFinite(value)) {
    log.warn('Sampling', `Invalid ${name}=${value}, using default=${fallback}`);
    return fallback;
  }
  if (value < min) {
    log.warn('Sampling', `${name}=${value} below minimum ${min}, clamping to ${min}`);
    return min;
  }
  if (value > max) {
    log.warn('Sampling', `${name}=${value} above maximum ${max}, clamping to ${max}`);
    return max;
  }
  return value;
}
