import { log } from '../debug/index.js';

/**
 * Known quantization levels and their typical minimum safe thresholds.
 * Thresholds tighter than these are likely to produce false negatives.
 */
const QUANTIZATION_PRECISION_FLOORS = {
  'q2k': 0.15,
  'q3k': 0.10,
  'q4k': 0.05,
  'q4_0': 0.05,
  'q4_1': 0.05,
  'q5k': 0.03,
  'q5_0': 0.03,
  'q5_1': 0.03,
  'q6k': 0.02,
  'q8_0': 0.01,
  'q8k': 0.01,
  'f16': 0.001,
  'f32': 0.0001,
};

/**
 * Validate verification thresholds against the model's quantization level.
 * Warns when the threshold appears too tight for the given quantization,
 * which would cause false verification failures.
 *
 * @param {object} options
 * @param {number} options.threshold - The verification threshold (e.g. max allowed drift).
 * @param {string|null} [options.quantization] - The model's quantization level (e.g. 'q4k', 'f16').
 * @param {string} [options.label] - Optional label for log messages.
 * @returns {{ valid: boolean, warning: string|null }}
 */
export function validateThresholdForPrecision(options = {}) {
  const { threshold, quantization, label } = options;

  if (threshold == null || typeof threshold !== 'number') {
    return { valid: true, warning: null };
  }

  if (!quantization || typeof quantization !== 'string') {
    return { valid: true, warning: null };
  }

  const normalizedQuant = quantization.toLowerCase().replace(/-/g, '_');
  const precisionFloor = QUANTIZATION_PRECISION_FLOORS[normalizedQuant];

  if (precisionFloor == null) {
    return { valid: true, warning: null };
  }

  if (threshold < precisionFloor) {
    const prefix = label ? `${label}: ` : '';
    const warning =
      `${prefix}Verification threshold ${threshold} is tighter than the expected precision ` +
      `floor ${precisionFloor} for quantization "${quantization}". ` +
      'This may cause false verification failures.';
    log.warn('verify-runner', warning);
    return { valid: false, warning };
  }

  return { valid: true, warning: null };
}
