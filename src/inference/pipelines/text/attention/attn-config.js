import { log } from '../../../../debug/index.js';

const ATTN_CONFIG_REQUIRED_FIELDS = Object.freeze([
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
  'queryScale',
  'queryKeyNormType',
  'queryKeyNormAxis',
]);

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
