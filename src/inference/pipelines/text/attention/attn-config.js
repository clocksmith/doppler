import { log } from '../../../../debug/index.js';

/**
 * Required fields that must not be undefined on every AttnConfig instance.
 * These are the fields destructured by both run.js and record.js attention
 * entry points. Optional fields (skipInputNorm, tokenIds, kernelPath,
 * disableRoPE) have defaults in the destructuring and are excluded.
 */
export const ATTN_CONFIG_REQUIRED_FIELDS = Object.freeze([
  'layerIdx',
  'numTokens',
  'isPrefill',
  'numHeads',
  'numKVHeads',
  'headDim',
  'hiddenSize',
  'rmsNormEps',
  'currentSeqLen',
  'activationDtype',
  'attnSoftcap',
  'queryPreAttnScalar',
]);

/**
 * Validate an AttnConfig object at construction time.
 *
 * Checks that all required fields are not undefined. Logs a warning per
 * missing field. Returns true when all required fields are present.
 *
 * @param {Record<string, unknown>} config - The attention config to validate.
 * @param {string} [label] - Diagnostic label (e.g., "L0" for layer 0).
 * @returns {boolean} True if all required fields are present.
 */
export function validateAttnConfig(config, label) {
  if (!config || typeof config !== 'object') {
    log.warn('Attention', `${label ?? 'attnConfig'}: config is null or not an object.`);
    return false;
  }
  let valid = true;
  for (const field of ATTN_CONFIG_REQUIRED_FIELDS) {
    if (config[field] === undefined) {
      log.warn(
        'Attention',
        `${label ?? 'attnConfig'}: required field "${field}" is undefined. ` +
        'This may cause unexpected behavior in the attention dispatch path.'
      );
      valid = false;
    }
  }
  return valid;
}
