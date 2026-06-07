const REQUIRED_RUNTIME_SAMPLING_FIELDS = [
  'temperature',
  'topP',
  'topK',
  'repetitionPenalty',
  'repetitionPenaltyWindow',
  'greedyThreshold',
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
      (value) => value >= 1
    ),
    repetitionPenalty: resolveSamplingNumber(
      'repetitionPenalty',
      opts?.repetitionPenalty,
      samplingDefaults.repetitionPenalty,
      (value) => value > 0
    ),
    repetitionPenaltyWindow: resolveSamplingInteger(
      'repetitionPenaltyWindow',
      undefined,
      samplingDefaults.repetitionPenaltyWindow,
      (value) => value >= 0
    ),
    greedyThreshold: resolveSamplingNumber(
      'greedyThreshold',
      undefined,
      samplingDefaults.greedyThreshold,
      (value) => value >= 0
    ),
  };
}

function resolveSamplingNumber(name, callValue, runtimeValue, validate) {
  const value = callValue === undefined ? runtimeValue : callValue;
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
