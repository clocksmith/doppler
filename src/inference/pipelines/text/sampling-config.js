const REQUIRED_RUNTIME_SAMPLING_FIELDS = [
  'temperature',
  'topP',
  'topK',
  'repetitionPenalty',
  'presencePenalty',
  'repetitionPenaltyWindow',
  'greedyThreshold',
  'suppressSpecialTokens',
  'suppressSpecialLikeTokens',
  'suppressTokenIds',
];

export function resolveSamplingConfig(opts, runtimeConfig) {
  const samplingDefaults = runtimeConfig?.inference?.sampling;
  if (!samplingDefaults || typeof samplingDefaults !== 'object' || Array.isArray(samplingDefaults)) {
    throw new Error('[Sampling] runtimeConfig.inference.sampling is required.');
  }
  for (const field of REQUIRED_RUNTIME_SAMPLING_FIELDS) {
    if (samplingDefaults[field] === undefined) {
      throw new Error(`[Sampling] runtimeConfig.inference.sampling.${field} is required.`);
    }
  }

  return {
    temperature: resolveSamplingNumber(
      'temperature',
      opts?.temperature,
      samplingDefaults.temperature,
      (value) => value >= 0
    ),
    topP: resolveSamplingNumber(
      'topP',
      opts?.topP,
      samplingDefaults.topP,
      (value) => value > 0 && value <= 1
    ),
    topK: resolveSamplingInteger(
      'topK',
      opts?.topK,
      samplingDefaults.topK,
      (value) => value >= 0
    ),
    repetitionPenalty: resolveSamplingNumber(
      'repetitionPenalty',
      opts?.repetitionPenalty,
      samplingDefaults.repetitionPenalty,
      (value) => value > 0
    ),
    presencePenalty: resolveSamplingNumber(
      'presencePenalty',
      opts?.presencePenalty,
      samplingDefaults.presencePenalty,
      (value) => value >= 0
    ),
    repetitionPenaltyWindow: resolveSamplingInteger(
      'repetitionPenaltyWindow',
      opts?.repetitionPenaltyWindow,
      samplingDefaults.repetitionPenaltyWindow,
      (value) => value >= 0
    ),
    greedyThreshold: resolveSamplingNumber(
      'greedyThreshold',
      undefined,
      samplingDefaults.greedyThreshold,
      (value) => value >= 0
    ),
    suppressSpecialTokens: resolveSamplingBoolean(
      'suppressSpecialTokens',
      opts?.suppressSpecialTokens === undefined ? samplingDefaults.suppressSpecialTokens : opts.suppressSpecialTokens
    ),
    suppressSpecialLikeTokens: resolveSamplingBoolean(
      'suppressSpecialLikeTokens',
      opts?.suppressSpecialLikeTokens === undefined ? samplingDefaults.suppressSpecialLikeTokens : opts.suppressSpecialLikeTokens
    ),
    suppressTokenIds: resolveSamplingTokenIdList(
      'suppressTokenIds',
      opts?.suppressTokenIds === undefined ? samplingDefaults.suppressTokenIds : opts.suppressTokenIds
    ),
  };
}

function resolveSamplingBoolean(name, runtimeValue) {
  if (runtimeValue === null) {
    throw new Error(`[Sampling] ${name} cannot be null.`);
  }
  if (typeof runtimeValue !== 'boolean') {
    throw new Error(`[Sampling] ${name} must be a boolean; got ${typeof runtimeValue}.`);
  }
  return runtimeValue;
}

function resolveSamplingTokenIdList(name, runtimeValue) {
  if (!Array.isArray(runtimeValue)) {
    throw new Error(`[Sampling] ${name} must be an array of token IDs.`);
  }
  return runtimeValue.map((tokenId, index) => {
    if (!Number.isInteger(tokenId) || tokenId < 0) {
      throw new Error(`[Sampling] ${name}[${index}] must be a non-negative integer token ID; got ${tokenId}.`);
    }
    return tokenId;
  });
}

function resolveSamplingNumber(name, callValue, runtimeValue, validate) {
  const value = callValue === undefined ? runtimeValue : callValue;
  if (Object.hasOwn(GENERATION_CONTRACT.options, name)) {
    validateGenerationField(name, value);
    return value;
  }
  if (value === null) {
    throw new Error(`[Sampling] ${name} cannot be null.`);
  }
  if (!Number.isFinite(value)) {
    throw new Error(`[Sampling] ${name} must be a finite number; got ${value}.`);
  }
  if (!validate(value)) {
    throw new Error(`[Sampling] ${name} is outside the configured range; got ${value}.`);
  }
  return value;
}

function resolveSamplingInteger(name, callValue, runtimeValue, validate) {
  const value = resolveSamplingNumber(name, callValue, runtimeValue, validate);
  if (Math.floor(value) !== value) {
    throw new Error(`[Sampling] ${name} must be an integer; got ${value}.`);
  }
  return value;
}
import { GENERATION_CONTRACT, validateGenerationField } from '../../../config/generation-contract.js';
