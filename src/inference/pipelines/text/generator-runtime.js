import { readBuffer } from '../../../memory/buffer-pool.js';
import { matmulCPU, rmsNormCPU } from './logits/index.js';
import { isGpuBufferInstance, isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
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

function resolveExplicitInputIds(inputIds, context) {
  if (inputIds === undefined) {
    return null;
  }
  if (inputIds === null) {
    throw new Error(`[Pipeline] ${context}: null is unsupported; omit the key or pass explicit token IDs.`);
  }
  if (!Array.isArray(inputIds) && !ArrayBuffer.isView(inputIds)) {
    throw new Error(
      `[Pipeline] ${context}: expected an array or typed array of token IDs, got ${typeof inputIds}.`
    );
  }
  return Array.from(inputIds, (value, index) => {
    if (!Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
      throw new Error(
        `[Pipeline] ${context}[${index}]: expected a non-negative integer token ID, got ${value}.`
      );
    }
    return value;
  });
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

  const runtimeOverride = state.runtimeConfig.inference.chatTemplate?.enabled;
  const fromRuntime = runtimeOverride == null
    ? undefined
    : resolveConfiguredValue(
      runtimeOverride,
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
    seed: resolveConfiguredValue(
      options.seed,
      undefined,
      'options.seed',
      (value) => Number.isFinite(value) && value >= 0
    ),
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
    inputIds: resolveExplicitInputIds(options.inputIds, 'options.inputIds'),
    embeddingOverrides: options.embeddingOverrides ?? null,
    embeddingInputSpan: options.__internalEmbeddingInputSpan ?? null,
    multimodalBidirectionalSpan: options.__internalMultimodalBidirectionalSpan ?? null,
  };
}

export function resolveGenerateOptions(state, options = {}) {
  const runtimeDefaults = state.runtimeConfig.inference;
  const samplingDefaults = runtimeDefaults.sampling;
  const generationDefaults = runtimeDefaults.generation;
  const executionPlan = resolveExecutionSessionPlan(state, options);

  return {
    seed: resolveConfiguredValue(
      options.seed,
      undefined,
      'options.seed',
      (value) => Number.isFinite(value) && value >= 0
    ),
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
    images: options.images ?? null,
    speculation: resolveSpeculationConfig(state, options),
    inputIds: resolveExplicitInputIds(options.inputIds, 'options.inputIds'),
    embeddingOverrides: options.embeddingOverrides ?? null,
    embeddingInputSpan: options.__internalEmbeddingInputSpan ?? null,
    multimodalBidirectionalSpan: options.__internalMultimodalBidirectionalSpan ?? null,
  };
}

function resolveSpeculationConfig(state, options) {
  const sessionSpeculation = state.runtimeConfig?.inference?.session?.speculation ?? null;
  const callSpeculation = options.speculation ?? null;
  if (!sessionSpeculation && !callSpeculation) return null;
  return {
    mode: callSpeculation?.mode ?? sessionSpeculation?.mode ?? 'none',
    tokens: callSpeculation?.tokens ?? sessionSpeculation?.tokens ?? 1,
    verify: callSpeculation?.verify ?? sessionSpeculation?.verify ?? 'greedy',
    threshold: callSpeculation?.threshold ?? sessionSpeculation?.threshold ?? null,
    rollbackOnReject: callSpeculation?.rollbackOnReject ?? sessionSpeculation?.rollbackOnReject ?? true,
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
    images: options.images ?? null,
    inputIds: resolveExplicitInputIds(options.inputIds, 'options.inputIds'),
    embeddingOverrides: options.embeddingOverrides ?? null,
    embeddingInputSpan: options.__internalEmbeddingInputSpan ?? null,
    multimodalBidirectionalSpan: options.__internalMultimodalBidirectionalSpan ?? null,
  };
}

export function resolvePrefillEmbeddingOptions(state, options = {}) {
  const postprocessor = state.modelConfig?.embeddingPostprocessor ?? null;
  const requestedEmbeddingMode = resolveConfiguredValue(
    options.embeddingMode,
    undefined,
    'options.embeddingMode',
    (value) => value === 'last' || value === 'mean'
  );
  if (postprocessor) {
    if (requestedEmbeddingMode !== undefined && requestedEmbeddingMode !== postprocessor.poolingMode) {
      throw new Error(
        `[Pipeline] options.embeddingMode="${requestedEmbeddingMode}" conflicts with ` +
        `manifest output.embeddingPostprocessor.poolingMode="${postprocessor.poolingMode}".`
      );
    }
    return {
      ...resolvePrefillOptions(state, options),
      embeddingMode: postprocessor.poolingMode,
    };
  }
  const modelType = typeof state.manifest?.modelType === 'string'
    ? state.manifest.modelType.toLowerCase()
    : '';
  const generationDefaults = state.runtimeConfig.inference.generation;
  // Embedding models default to 'mean' pooling — this is a model-category behavior,
  // not a model-family identity check. Ideally embedding conversion configs would set
  // generation.embeddingMode='mean' in their runtime config; the modelType fallback
  // provides this default for manifests that predate runtime-profile embedding mode.
  const defaultEmbeddingMode = modelType === 'embedding'
    ? 'mean'
    : generationDefaults.embeddingMode;
  return {
    ...resolvePrefillOptions(state, options),
    embeddingMode: requestedEmbeddingMode ?? defaultEmbeddingMode,
  };
}

export function resolveAdvanceEmbeddingMode(state, options = {}) {
  if (state.modelConfig?.embeddingPostprocessor) {
    throw new Error(
      '[Pipeline] advanceWithTokenAndEmbedding is unsupported when manifest output.embeddingPostprocessor is enabled.'
    );
  }
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

function decodeFloatWeights(data, dtype, expectedLength, label) {
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
  } else if (isGpuBufferInstance(finalNorm)) {
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
  config,
  embeddingPostprocessor = null
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

  const postprocessorConfig = config?.embeddingPostprocessor ?? null;
  const resolvedEmbeddingMode = postprocessorConfig?.poolingMode ?? embeddingMode;
  if (postprocessorConfig && embeddingMode !== resolvedEmbeddingMode) {
    throw new Error(
      `[Pipeline] embeddingMode "${embeddingMode}" conflicts with manifest output.embeddingPostprocessor.poolingMode="${resolvedEmbeddingMode}".`
    );
  }

  let pooled;
  if (resolvedEmbeddingMode === 'last') {
    pooled = applyFinalNorm(numTokens - 1);
  } else if (resolvedEmbeddingMode === 'mean') {
    pooled = new Float32Array(hiddenSize);
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
  } else {
    throw new Error(`prefillWithEmbedding: unsupported embeddingMode "${resolvedEmbeddingMode}" (expected "last" or "mean")`);
  }

  if (!postprocessorConfig) {
    return pooled;
  }
  if (!embeddingPostprocessor) {
    throw new Error('[Pipeline] Embedding postprocessor weights are missing for this manifest.');
  }

  let current = pooled;
  for (let i = 0; i < embeddingPostprocessor.projections.length; i++) {
    const projection = embeddingPostprocessor.projections[i];
    if (current.length !== projection.inputSize) {
      throw new Error(
        `[Pipeline] Embedding postprocessor projection ${i} expected inputSize=${projection.inputSize}, got ${current.length}.`
      );
    }
    if (!(projection.weight instanceof Float32Array) || projection.weight.length !== (projection.outputSize * projection.inputSize)) {
      throw new Error(
        `[Pipeline] Embedding postprocessor projection ${i} has invalid weight shape for ${projection.outputSize}x${projection.inputSize}.`
      );
    }
    if (projection.activation !== 'identity') {
      throw new Error(
        `[Pipeline] Unsupported embedding postprocessor activation "${projection.activation}" at projection ${i}.`
      );
    }
    const projected = matmulCPU(current, projection.weight, 1, projection.outputSize, projection.inputSize, 'row');
    if (projection.bias) {
      if (!(projection.bias instanceof Float32Array) || projection.bias.length !== projection.outputSize) {
        throw new Error(
          `[Pipeline] Embedding postprocessor projection ${i} bias length mismatch: expected=${projection.outputSize}.`
        );
      }
      for (let j = 0; j < projected.length; j++) {
        projected[j] += projection.bias[j];
      }
    }
    current = projected;
  }

  if (embeddingPostprocessor.normalize === 'l2') {
    let sumSq = 0;
    for (let i = 0; i < current.length; i++) {
      sumSq += current[i] * current[i];
    }
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      const invNorm = 1 / norm;
      for (let i = 0; i < current.length; i++) {
        current[i] *= invNorm;
      }
    }
  }

  return current;
}
