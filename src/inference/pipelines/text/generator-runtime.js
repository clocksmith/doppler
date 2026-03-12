import { readBuffer } from '../../../memory/buffer-pool.js';
import { rmsNormCPU } from './logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { decodeReadback } from './debug-utils/index.js';
import { resolveExecutionSessionPlan } from './execution-plan.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

const UNKNOWN_TOKENIZER_VOCAB_SIZE = 'unknown';
const DEFAULT_DTYPE = 'f32';

function resolveConfiguredValue(value, defaultValue, context, validate) {
  if (value === undefined) {
    return defaultValue;
  }
  if (value === null) {
    throw new Error(`[Pipeline] ${context}: null is unsupported; omit the key or pass an explicit value.`);
  }
  if (validate && !validate(value)) {
    throw new Error(`[Pipeline] ${context}: invalid value "${value}".`);
  }
  return value;
}

function readTokenizerVocabSize(tok) {
  const tokenizerVocabSize = tok?.getVocabSize?.();
  return typeof tokenizerVocabSize === 'number' && Number.isFinite(tokenizerVocabSize)
    ? tokenizerVocabSize
    : null;
}

function readOptionalTokenizerText(tok, tokenId) {
  if (!tok || typeof tok.decode !== 'function') {
    return null;
  }
  try {
    return tok.decode([tokenId], false, false);
  } catch {
    return null;
  }
}

export function assertTokenIdsInRange(state, tokenIds, context = 'encode') {
  const vocabSize = state?.modelConfig?.vocabSize;
  if (!Array.isArray(tokenIds)) {
    throw new Error(`[Tokenizer] ${context}: expected tokenIds array, got ${typeof tokenIds}`);
  }
  if (!Number.isFinite(vocabSize) || vocabSize <= 0) {
    throw new Error(`[Tokenizer] ${context}: invalid model vocabSize=${vocabSize}`);
  }

  let firstBadIdx = -1;
  let firstBadId = -1;
  let maxId = -1;
  let badCount = 0;
  for (let i = 0; i < tokenIds.length; i++) {
    const id = tokenIds[i];
    if (!Number.isFinite(id) || id < 0 || id >= vocabSize) {
      badCount++;
      if (firstBadIdx < 0) {
        firstBadIdx = i;
        firstBadId = id;
      }
    }
    if (Number.isFinite(id) && id > maxId) maxId = id;
  }
  if (badCount === 0) return;

  const tok = state?.tokenizer;
  const tokenizerVocabSize = readTokenizerVocabSize(tok);
  const badText = readOptionalTokenizerText(tok, firstBadId);
  const safeTokenizerVocabSize = tokenizerVocabSize === null
    ? UNKNOWN_TOKENIZER_VOCAB_SIZE
    : tokenizerVocabSize;

  throw new Error(
    `[Tokenizer] ${context}: token id out of range for model vocab. ` +
    `modelVocabSize=${vocabSize}, tokenizerVocabSize=${safeTokenizerVocabSize}, ` +
    `badCount=${badCount}/${tokenIds.length}, firstBadIdx=${firstBadIdx}, firstBadId=${firstBadId}` +
    (badText === null ? '' : ` ("${badText}")`) +
    `, maxId=${maxId}. ` +
    'This will poison GPU embedding gather (NaNs). Fix by re-converting the model or aligning tokenizer.json IDs to embedding/LM-head shapes.'
  );
}

export function assertTokenIdInRange(state, tokenId, context = 'token') {
  const vocabSize = state?.modelConfig?.vocabSize;
  if (!Number.isFinite(vocabSize) || vocabSize <= 0) {
    throw new Error(`[Tokenizer] ${context}: invalid model vocabSize=${vocabSize}`);
  }
  if (!Number.isFinite(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
    const tok = state?.tokenizer;
    const tokenizerVocabSize = readTokenizerVocabSize(tok);
    const safeTokenizerVocabSize = tokenizerVocabSize === null
      ? UNKNOWN_TOKENIZER_VOCAB_SIZE
      : tokenizerVocabSize;
    throw new Error(
      `[Tokenizer] ${context}: tokenId=${tokenId} out of range (modelVocabSize=${vocabSize}, tokenizerVocabSize=${safeTokenizerVocabSize}).`
    );
  }
}

function resolveChatTemplateEnabled(state, options) {
  const fromOptions = resolveConfiguredValue(
    options.useChatTemplate,
    undefined,
    'options.useChatTemplate',
    (value) => typeof value === 'boolean'
  );
  if (fromOptions !== undefined) {
    return fromOptions;
  }

  const fromRuntime = resolveConfiguredValue(
    state.runtimeConfig.inference.chatTemplate?.enabled,
    undefined,
    'state.runtimeConfig.inference.chatTemplate.enabled',
    (value) => typeof value === 'boolean'
  );
  if (fromRuntime !== undefined) {
    return fromRuntime;
  }

  const fromModel = resolveConfiguredValue(
    state.modelConfig?.chatTemplateEnabled,
    undefined,
    'state.modelConfig.chatTemplateEnabled',
    (value) => typeof value === 'boolean'
  );
  if (fromModel !== undefined) {
    return fromModel;
  }

  return false;
}

export function resolveStepOptions(state, options = {}) {
  const runtimeDefaults = state.runtimeConfig.inference;
  const samplingDefaults = runtimeDefaults.sampling;
  const executionPlan = resolveExecutionSessionPlan(state, options);

  return {
    temperature: resolveConfiguredValue(options.temperature, samplingDefaults.temperature, 'options.temperature'),
    topP: resolveConfiguredValue(options.topP, samplingDefaults.topP, 'options.topP'),
    topK: resolveConfiguredValue(options.topK, samplingDefaults.topK, 'options.topK'),
    repetitionPenalty: resolveConfiguredValue(
      options.repetitionPenalty,
      samplingDefaults.repetitionPenalty,
      'options.repetitionPenalty'
    ),
    debug: resolveConfiguredValue(options.debug, state.debug, 'options.debug', (value) => typeof value === 'boolean'),
    debugLayers: options.debugLayers,
    profile: resolveConfiguredValue(options.profile, runtimeDefaults.generation.profile, 'options.profile'),
    disableCommandBatching: executionPlan.disableCommandBatching,
    disableMultiTokenDecode: executionPlan.disableMultiTokenDecode,
    batchSize: executionPlan.batchSize,
    stopCheckMode: executionPlan.stopCheckMode,
    executionPlan,
  };
}

export function resolveGenerateOptions(state, options = {}) {
  const runtimeDefaults = state.runtimeConfig.inference;
  const samplingDefaults = runtimeDefaults.sampling;
  const generationDefaults = runtimeDefaults.generation;
  const executionPlan = resolveExecutionSessionPlan(state, options);

  return {
    maxTokens: executionPlan.maxTokens,
    temperature: resolveConfiguredValue(options.temperature, samplingDefaults.temperature, 'options.temperature'),
    topP: resolveConfiguredValue(options.topP, samplingDefaults.topP, 'options.topP'),
    topK: resolveConfiguredValue(options.topK, samplingDefaults.topK, 'options.topK'),
    repetitionPenalty: resolveConfiguredValue(
      options.repetitionPenalty,
      samplingDefaults.repetitionPenalty,
      'options.repetitionPenalty'
    ),
    stopSequences: resolveConfiguredValue(options.stopSequences, [], 'options.stopSequences', Array.isArray),
    useSpeculative: resolveConfiguredValue(
      options.useSpeculative,
      generationDefaults.useSpeculative,
      'options.useSpeculative',
      (value) => typeof value === 'boolean'
    ),
    useChatTemplate: resolveChatTemplateEnabled(state, options),
    debug: resolveConfiguredValue(options.debug, state.debug, 'options.debug', (value) => typeof value === 'boolean'),
    debugLayers: options.debugLayers,
    profile: resolveConfiguredValue(options.profile, generationDefaults.profile, 'options.profile'),
    benchmark: resolveConfiguredValue(options.benchmark, generationDefaults.benchmark, 'options.benchmark'),
    disableCommandBatching: executionPlan.disableCommandBatching,
    disableMultiTokenDecode: executionPlan.disableMultiTokenDecode,
    batchSize: executionPlan.batchSize,
    stopCheckMode: executionPlan.stopCheckMode,
    executionPlan,
  };
}

export function resolvePrefillOptions(state, options = {}) {
  const generationDefaults = state.runtimeConfig.inference.generation;
  const executionPlan = resolveExecutionSessionPlan(state, options);
  return {
    useChatTemplate: resolveChatTemplateEnabled(state, options),
    debug: resolveConfiguredValue(options.debug, state.debug, 'options.debug', (value) => typeof value === 'boolean'),
    debugLayers: options.debugLayers,
    profile: resolveConfiguredValue(options.profile, generationDefaults.profile, 'options.profile'),
    disableCommandBatching: executionPlan.disableCommandBatching,
    disableMultiTokenDecode: executionPlan.disableMultiTokenDecode,
    executionPlan,
  };
}

export function resolvePrefillEmbeddingOptions(state, options = {}) {
  const modelType = typeof state.manifest?.modelType === 'string'
    ? state.manifest.modelType.toLowerCase()
    : '';
  const generationDefaults = state.runtimeConfig.inference.generation;
  // Embedding models default to 'mean' pooling — this is a model-category behavior,
  // not a model-family identity check. Ideally embedding model presets would set
  // generation.embeddingMode='mean' in their runtime config; the modelType fallback
  // provides this default for manifests that predate runtime-preset embedding mode.
  const defaultEmbeddingMode = modelType === 'embedding'
    ? 'mean'
    : generationDefaults.embeddingMode;
  return {
    ...resolvePrefillOptions(state, options),
    embeddingMode: resolveConfiguredValue(options.embeddingMode, defaultEmbeddingMode, 'options.embeddingMode'),
  };
}

export function resolveAdvanceEmbeddingMode(state, options = {}) {
  const modelType = typeof state.manifest?.modelType === 'string'
    ? state.manifest.modelType.toLowerCase()
    : '';
  // See resolvePrefillEmbeddingOptions for embedding-model pooling rationale.
  const configuredMode = state.runtimeConfig.inference.generation.embeddingMode;
  return resolveConfiguredValue(
    options.embeddingMode,
    modelType === 'embedding' ? 'mean' : configuredMode,
    'options.embeddingMode',
    (value) => value === 'last' || value === 'mean'
  );
}

function resolveFloatDtypeFromAlias(dtype) {
  const normalized = typeof dtype === 'string' ? dtype.trim().toLowerCase() : '';
  if (!normalized) return DEFAULT_DTYPE;
  return selectRuleValue('inference', 'dtype', 'dtypeFromAlias', {
    dtype: normalized,
    fallback: DEFAULT_DTYPE,
  });
}

export function resolveFloatDtypeFromByteSize(totalBytes, expectedLength) {
  if (!Number.isFinite(totalBytes) || totalBytes <= 0 || !Number.isFinite(expectedLength) || expectedLength <= 0) {
    return DEFAULT_DTYPE;
  }
  const bytesPerElement = totalBytes / expectedLength;
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromBytesOrFallback', {
    bytesPerElement,
    fallback: DEFAULT_DTYPE,
  });
}

export function decodeFloatWeights(data, dtype, expectedLength, label) {
  const decodeDtype = resolveFloatDtypeFromAlias(dtype);
  const decoded = decodeReadback(data, decodeDtype);
  if (decoded.length !== expectedLength) {
    throw new Error(
      `[Pipeline] ${label} length mismatch: expected=${expectedLength}, got=${decoded.length}`
    );
  }
  return decoded;
}

export async function getFinalNormWeights(state) {
  const hiddenSize = state.modelConfig.hiddenSize;
  const finalNorm = state.weights.get('final_norm');
  if (!finalNorm) {
    throw new Error('[Pipeline] final_norm weight is missing; cannot extract embedding.');
  }

  let weights;

  if (finalNorm instanceof Float32Array) {
    weights = finalNorm;
  } else if (isCpuWeightBuffer(finalNorm)) {
    const dtype = resolveFloatDtypeFromAlias(finalNorm.dtype);
    const data = finalNorm.data;
    if (!(data instanceof Float32Array) && !ArrayBuffer.isView(data)) {
      throw new Error('[Pipeline] final_norm CPU weight buffer has unsupported data type.');
    }
    const bytes = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    weights = decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
  } else if (isWeightBuffer(finalNorm)) {
    const dtypeValue = typeof finalNorm.dtype === 'string' ? finalNorm.dtype.trim().toLowerCase() : '';
    const dtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtypeAlias', {
      dtype: dtypeValue === '' ? undefined : dtypeValue,
      fallback: DEFAULT_DTYPE,
    });
    const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
    const readSize = hiddenSize * bytesPerElement;
    const data = await readBuffer(finalNorm.buffer, readSize);
    if (data.byteLength === 0) {
      throw new Error('[Pipeline] final_norm readback returned empty buffer.');
    }
    weights = decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
  } else if (finalNorm instanceof GPUBuffer) {
    const dtype = resolveFloatDtypeFromByteSize(finalNorm.size, hiddenSize);
    const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
    const readSize = hiddenSize * bytesPerElement;
    const data = await readBuffer(finalNorm, readSize);
    if (data.byteLength === 0) {
      throw new Error('[Pipeline] final_norm readback returned empty buffer.');
    }
    weights = decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
  } else if (ArrayBuffer.isView(finalNorm)) {
    const view = finalNorm;
    const dtype = resolveFloatDtypeFromByteSize(view.byteLength, hiddenSize);
    const bytes = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
    weights = decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
  } else {
    throw new Error('[Pipeline] final_norm weight has unsupported type.');
  }
  if (!(weights instanceof Float32Array) || weights.length !== hiddenSize) {
    const reportedLength = weights === undefined || weights === null ? UNKNOWN_TOKENIZER_VOCAB_SIZE : weights.length;
    throw new Error(
      `[Pipeline] final_norm length mismatch: expected=${hiddenSize}, got=${reportedLength}`
    );
  }
  return weights;
}

export function extractEmbeddingFromHidden(
  hiddenStates,
  numTokens,
  hiddenSize,
  embeddingMode,
  finalNormWeights,
  config
) {
  const expectedLength = numTokens * hiddenSize;
  if (hiddenStates.length !== expectedLength) {
    throw new Error(
      `[Pipeline] Hidden state length mismatch for embedding extraction: expected=${expectedLength}, got=${hiddenStates.length}`
    );
  }

  const applyFinalNorm = (tokenIndex) => {
    const offset = tokenIndex * hiddenSize;
    const tokenHidden = hiddenStates.subarray(offset, offset + hiddenSize);
    return rmsNormCPU(
      tokenHidden,
      finalNormWeights,
      config.rmsNormEps,
      config.rmsNormWeightOffset
    );
  };

  if (embeddingMode === 'last') {
    return applyFinalNorm(numTokens - 1);
  }

  if (embeddingMode === 'mean') {
    const pooled = new Float32Array(hiddenSize);
    for (let t = 0; t < numTokens; t++) {
      const tokenEmbedding = applyFinalNorm(t);
      for (let i = 0; i < hiddenSize; i++) {
        pooled[i] += tokenEmbedding[i];
      }
    }
    const invTokens = numTokens > 0 ? (1 / numTokens) : 1;
    for (let i = 0; i < hiddenSize; i++) {
      pooled[i] *= invTokens;
    }
    return pooled;
  }

  throw new Error(`prefillWithEmbedding: unsupported embeddingMode "${embeddingMode}" (expected "last" or "mean")`);
}
