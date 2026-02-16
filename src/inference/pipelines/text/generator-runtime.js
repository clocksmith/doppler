import { readBuffer } from '../../../memory/buffer-pool.js';
import { rmsNormCPU } from './logits.js';
import { isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { decodeReadback } from './debug-utils.js';

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
  const tokenizerVocabSize = tok?.getVocabSize?.() ?? null;
  let badText = null;
  try {
    badText = tok?.decode?.([firstBadId], false, false) ?? null;
  } catch {
    badText = null;
  }

  throw new Error(
    `[Tokenizer] ${context}: token id out of range for model vocab. ` +
    `modelVocabSize=${vocabSize}, tokenizerVocabSize=${tokenizerVocabSize ?? 'unknown'}, ` +
    `badCount=${badCount}/${tokenIds.length}, firstBadIdx=${firstBadIdx}, firstBadId=${firstBadId}` +
    (badText ? ` ("${badText}")` : '') +
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
    const tokenizerVocabSize = tok?.getVocabSize?.() ?? null;
    throw new Error(
      `[Tokenizer] ${context}: tokenId=${tokenId} out of range (modelVocabSize=${vocabSize}, tokenizerVocabSize=${tokenizerVocabSize ?? 'unknown'}).`
    );
  }
}

function resolveChatTemplateEnabled(state, options) {
  return options.useChatTemplate
    ?? state.runtimeConfig.inference.chatTemplate?.enabled
    ?? state.modelConfig?.chatTemplateEnabled
    ?? false;
}

export function resolveStepOptions(state, options = {}) {
  const runtimeDefaults = state.runtimeConfig.inference;
  const samplingDefaults = runtimeDefaults.sampling;
  const batchingDefaults = runtimeDefaults.batching;
  const generationDefaults = runtimeDefaults.generation;

  return {
    temperature: options.temperature ?? samplingDefaults.temperature,
    topP: options.topP ?? samplingDefaults.topP,
    topK: options.topK ?? samplingDefaults.topK,
    repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
    debug: options.debug ?? state.debug,
    debugLayers: options.debugLayers,
    profile: options.profile ?? generationDefaults.profile,
    disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
    disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
    batchSize: options.batchSize ?? batchingDefaults.batchSize,
    stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
  };
}

export function resolveGenerateOptions(state, options = {}) {
  const runtimeDefaults = state.runtimeConfig.inference;
  const samplingDefaults = runtimeDefaults.sampling;
  const batchingDefaults = runtimeDefaults.batching;
  const generationDefaults = runtimeDefaults.generation;

  return {
    maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
    temperature: options.temperature ?? samplingDefaults.temperature,
    topP: options.topP ?? samplingDefaults.topP,
    topK: options.topK ?? samplingDefaults.topK,
    repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
    stopSequences: options.stopSequences ?? [],
    useSpeculative: options.useSpeculative ?? generationDefaults.useSpeculative,
    useChatTemplate: resolveChatTemplateEnabled(state, options),
    debug: options.debug ?? state.debug,
    debugLayers: options.debugLayers,
    profile: options.profile ?? generationDefaults.profile,
    benchmark: options.benchmark ?? generationDefaults.benchmark,
    disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
    disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
    batchSize: options.batchSize ?? batchingDefaults.batchSize,
    stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
  };
}

export function resolvePrefillOptions(state, options = {}) {
  const generationDefaults = state.runtimeConfig.inference.generation;
  return {
    useChatTemplate: resolveChatTemplateEnabled(state, options),
    debug: options.debug ?? state.debug,
    debugLayers: options.debugLayers,
    profile: options.profile ?? generationDefaults.profile,
    disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
    disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
  };
}

export function resolvePrefillEmbeddingOptions(state, options = {}) {
  const modelType = String(state.manifest?.modelType || '').toLowerCase();
  const generationDefaults = state.runtimeConfig.inference.generation;
  const defaultEmbeddingMode = modelType === 'embedding'
    ? 'mean'
    : generationDefaults.embeddingMode;
  return {
    ...resolvePrefillOptions(state, options),
    embeddingMode: options.embeddingMode ?? defaultEmbeddingMode,
  };
}

export function resolveAdvanceEmbeddingMode(state, options = {}) {
  const modelType = String(state.manifest?.modelType || '').toLowerCase();
  const configuredMode = state.runtimeConfig.inference.generation.embeddingMode;
  return options.embeddingMode ?? (modelType === 'embedding' ? 'mean' : configuredMode);
}

export function resolveFloatDtypeFromByteSize(totalBytes, expectedLength, fallback = 'f32') {
  if (!Number.isFinite(totalBytes) || totalBytes <= 0 || !Number.isFinite(expectedLength) || expectedLength <= 0) {
    return fallback;
  }
  const bytesPerElement = totalBytes / expectedLength;
  if (Math.abs(bytesPerElement - 2) < 0.5) return 'f16';
  if (Math.abs(bytesPerElement - 4) < 0.5) return 'f32';
  return bytesPerElement < 3 ? 'f16' : 'f32';
}

export function decodeFloatWeights(data, dtype, expectedLength, label) {
  const decodeDtype = dtype === 'bf16'
    ? 'bf16'
    : (dtype === 'f16' ? 'f16' : 'f32');
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
    const dtype = finalNorm.dtype === 'bf16' ? 'bf16' : (finalNorm.dtype === 'f16' ? 'f16' : 'f32');
    const data = finalNorm.data;
    if (!(data instanceof Float32Array) && !ArrayBuffer.isView(data)) {
      throw new Error('[Pipeline] final_norm CPU weight buffer has unsupported data type.');
    }
    const bytes = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    weights = decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
  } else if (isWeightBuffer(finalNorm)) {
    const dtype = finalNorm.dtype === 'bf16' ? 'bf16' : (finalNorm.dtype === 'f16' ? 'f16' : 'f32');
    const bytesPerElement = dtype === 'f16' || dtype === 'bf16' ? 2 : 4;
    const readSize = hiddenSize * bytesPerElement;
    const data = await readBuffer(finalNorm.buffer, readSize);
    if (data.byteLength === 0) {
      throw new Error('[Pipeline] final_norm readback returned empty buffer.');
    }
    weights = decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
  } else if (finalNorm instanceof GPUBuffer) {
    const dtype = resolveFloatDtypeFromByteSize(finalNorm.size, hiddenSize, 'f32');
    const bytesPerElement = dtype === 'f16' ? 2 : 4;
    const readSize = hiddenSize * bytesPerElement;
    const data = await readBuffer(finalNorm, readSize);
    if (data.byteLength === 0) {
      throw new Error('[Pipeline] final_norm readback returned empty buffer.');
    }
    weights = decodeFloatWeights(data, dtype, hiddenSize, 'final_norm');
  } else if (ArrayBuffer.isView(finalNorm)) {
    const view = finalNorm;
    const dtype = resolveFloatDtypeFromByteSize(view.byteLength, hiddenSize, 'f32');
    const bytes = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
    weights = decodeFloatWeights(bytes, dtype, hiddenSize, 'final_norm');
  } else {
    throw new Error('[Pipeline] final_norm weight has unsupported type.');
  }
  if (!(weights instanceof Float32Array) || weights.length !== hiddenSize) {
    throw new Error(
      `[Pipeline] final_norm length mismatch: expected=${hiddenSize}, got=${weights?.length ?? 'unknown'}`
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
