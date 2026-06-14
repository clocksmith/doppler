const REQUIRED_POSITIVE_INTEGER_FIELDS = [
  'canvasLength',
  'maxDenoisingSteps',
  'maxNewTokens',
];

const REQUIRED_NON_NEGATIVE_NUMBER_FIELDS = [
  'confidenceThreshold',
  'tMin',
  'tMax',
];

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function readPositiveInteger(source, field, label) {
  const value = source[field];
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label}.${field} must be a positive integer.`);
  }
  return value;
}

function readNonNegativeNumber(source, field, label) {
  const value = source[field];
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label}.${field} must be a non-negative finite number.`);
  }
  return value;
}

function readNonNegativeInteger(source, field, label) {
  const value = source[field];
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label}.${field} must be a non-negative integer.`);
  }
  return value;
}

function readPositiveNumber(source, field, label) {
  const value = source[field];
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label}.${field} must be a positive finite number.`);
  }
  return value;
}

function readNullableTokenId(source, field, label) {
  const value = source[field];
  if (value === null) return null;
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label}.${field} must be null or a non-negative integer.`);
  }
  return value;
}

function readTokenId(source, field, label) {
  const value = source[field];
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label}.${field} must be a non-negative integer.`);
  }
  return value;
}

function readEosTokenIds(source, label) {
  const value = source.eosTokenIds;
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label}.eosTokenIds must be a non-empty array.`);
  }
  for (const tokenId of value) {
    if (!Number.isInteger(tokenId) || tokenId < 0) {
      throw new Error(`${label}.eosTokenIds must contain only non-negative integers.`);
    }
  }
  return [...value];
}

function readRouterContract(source, label) {
  const router = requireObject(source.router, `${label}.router`);
  for (const field of ['scaleHiddenStates', 'normalizeTopK', 'perExpertScale']) {
    if (typeof router[field] !== 'boolean') {
      throw new Error(`${label}.router.${field} must be boolean.`);
    }
  }
  return {
    scaleHiddenStates: router.scaleHiddenStates,
    normalizeTopK: router.normalizeTopK,
    perExpertScale: router.perExpertScale,
  };
}

export function parseDiffusionGemmaConfig(manifest) {
  const modelId = manifest?.modelId ?? 'unknown';
  const modelType = typeof manifest?.modelType === 'string'
    ? manifest.modelType.trim().toLowerCase()
    : '';
  if (modelType !== 'diffusion_gemma') {
    throw new Error(`Manifest "${modelId}" must declare modelType="diffusion_gemma".`);
  }

  const contract = requireObject(
    manifest?.inference?.diffusionGemma,
    `Manifest "${modelId}" inference.diffusionGemma`
  );
  const label = `Manifest "${modelId}" inference.diffusionGemma`;

  const parsed = {};
  for (const field of REQUIRED_POSITIVE_INTEGER_FIELDS) {
    parsed[field] = readPositiveInteger(contract, field, label);
  }
  for (const field of REQUIRED_NON_NEGATIVE_NUMBER_FIELDS) {
    parsed[field] = readNonNegativeNumber(contract, field, label);
  }
  parsed.stabilityThreshold = readNonNegativeInteger(contract, 'stabilityThreshold', label);
  parsed.entropyBound = readPositiveNumber(contract, 'entropyBound', label);
  if (contract.tMax <= contract.tMin) {
    throw new Error(`${label}.tMax must be greater than tMin.`);
  }
  if (contract.decoderCacheMode !== 'encoder_kv_readonly_canvas_concat') {
    throw new Error(
      `${label}.decoderCacheMode must be "encoder_kv_readonly_canvas_concat".`
    );
  }
  if (typeof contract.selfConditioning !== 'boolean') {
    throw new Error(`${label}.selfConditioning must be boolean.`);
  }

  const vocabSize = Number(manifest?.architecture?.vocabSize);
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new Error(`Manifest "${modelId}" architecture.vocabSize must be a positive integer.`);
  }

  return {
    ...parsed,
    vocabSize,
    padTokenId: readTokenId(contract, 'padTokenId', label),
    eosTokenIds: readEosTokenIds(contract, label),
    boiTokenId: readNullableTokenId(contract, 'boiTokenId', label),
    eoiTokenId: readNullableTokenId(contract, 'eoiTokenId', label),
    imageTokenId: readNullableTokenId(contract, 'imageTokenId', label),
    selfConditioning: contract.selfConditioning,
    decoderCacheMode: contract.decoderCacheMode,
    router: readRouterContract(contract, label),
  };
}
