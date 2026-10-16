import { mergeRecordOpLabelCounts } from './recorder-stats.js';
export { mergeRecordOpLabelCounts } from './recorder-stats.js';
import '../../../../gpu/device.js'; // Preserve declared device initialization side effects.
import { releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import { trace, isTraceEnabled } from '../../../../debug/index.js';
import {
  isWeightBuffer,
  isGpuBufferInstance,
  isSplitWeightBuffer,
  getWeightDtype,
  getLayout,
} from '../../../../gpu/weight-buffer.js';
import { decodeReadback, getLogitsHealth } from '../debug-utils/index.js';
import { tensorElementBytes } from '../../../../gpu/tensor.js';
import {
  getWeightBuffer as getPipelineWeightBuffer,
  getNormWeightBuffer as getPipelineNormWeightBuffer,
} from '../weights.js';

export const SPECIAL_LIKE_TOKEN_RE = /^(<pad>|<unused\d*>|<eos>|<bos>|<s>|<\/s>|\[PAD\]|\[UNK\]|\[SEP\]|\[CLS\]|<[^>\n]{1,32}>)$/i;

export const FINITENESS_RESET_WORDS = new Uint32Array(4);

export const tokenizerSuppressionCache = new WeakMap();

export const PREFILL_CHUNK_SUBMIT_MODES = new Set(['sync', 'async']);

export function recordPrefillRecorderStats(state, recorder) {
  if (!recorder || typeof recorder.getStats !== 'function') {
    return;
  }
  const recordStats = recorder.getStats();
  state.stats.prefillRecordOps = (state.stats.prefillRecordOps ?? 0) + recordStats.opCount;
  state.stats.prefillRecordPasses = (state.stats.prefillRecordPasses ?? 0) + recordStats.computePassCount;
  state.stats.prefillRecordOpLabels = mergeRecordOpLabelCounts(
    state.stats.prefillRecordOpLabels,
    recordStats.opLabelCounts
  );
}

export function resolvePrefillChunkSubmitMode(runtimeConfig, modelConfig) {
  const runtimeSubmit = runtimeConfig?.inference?.session?.prefillChunkSubmitMode;
  const manifestSubmit = modelConfig?.sessionSettings?.prefillChunkSubmitMode;
  const submit = (runtimeSubmit !== undefined && runtimeSubmit !== null)
    ? runtimeSubmit
    : manifestSubmit;
  if (submit === undefined || submit === null) {
    throw new Error('[Pipeline] runtime.inference.session.prefillChunkSubmitMode is required.');
  }
  if (!PREFILL_CHUNK_SUBMIT_MODES.has(submit)) {
    throw new Error(
      `[Pipeline] runtime.inference.session.prefillChunkSubmitMode must be "sync" or "async"; got "${String(submit)}".`
    );
  }
  return submit;
}

export function getTokenizerSuppressionCache(tokenizer) {
  let cache = tokenizerSuppressionCache.get(tokenizer);
  if (!cache) {
    cache = new Map();
    tokenizerSuppressionCache.set(tokenizer, cache);
  }
  return cache;
}

export function decodeSingleTokenForSuppression(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== 'function') {
    return '';
  }
  try {
    return tokenizer.decode([tokenId], false, false);
  } catch {
    return '';
  }
}

export function collectTokenizerSuppressedTokenIds(tokenizer, vocabSize, samplingConfig) {
  if (!tokenizer || !Number.isInteger(vocabSize) || vocabSize < 1) {
    return [];
  }
  if (samplingConfig.suppressSpecialTokens !== true && samplingConfig.suppressSpecialLikeTokens !== true) {
    return [];
  }
  const cache = getTokenizerSuppressionCache(tokenizer);
  const cacheKey = [
    vocabSize,
    samplingConfig.suppressSpecialTokens === true ? 'special' : 'plain',
    samplingConfig.suppressSpecialLikeTokens === true ? 'specialLike' : 'literal',
  ].join(':');
  const cached = cache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const tokenIds = [];
  for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
    const tokenizerSpecial = samplingConfig.suppressSpecialTokens === true
      && typeof tokenizer.isSpecialToken === 'function'
      && tokenizer.isSpecialToken(tokenId);
    const specialLike = samplingConfig.suppressSpecialLikeTokens === true
      && SPECIAL_LIKE_TOKEN_RE.test(decodeSingleTokenForSuppression(tokenizer, tokenId).trim());
    if (tokenizerSpecial || specialLike) {
      tokenIds.push(tokenId);
    }
  }
  cache.set(cacheKey, tokenIds);
  return tokenIds;
}

export function resolveSuppressedSamplingTokenIds(state, samplingConfig) {
  const vocabSize = state.modelConfig?.vocabSize;
  const suppressed = new Set(samplingConfig.suppressTokenIds);
  const stopTokenIds = new Set(state.modelConfig?.stopTokenIds ?? []);
  const eosToken = state.tokenizer?.getSpecialTokens?.()?.eos;
  if (Number.isInteger(eosToken)) {
    stopTokenIds.add(eosToken);
  }
  for (const tokenId of collectTokenizerSuppressedTokenIds(state.tokenizer, vocabSize, samplingConfig)) {
    if (!stopTokenIds.has(tokenId)) {
      suppressed.add(tokenId);
    }
  }
  return [...suppressed];
}

export function normalizeSelectedLogitTokenIds(value, vocabSize, label) {
  if (value == null) {
    return null;
  }
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new Error(`${label} must be an array or typed array.`);
  }
  const tokenIds = Array.from(value, (entry, index) => {
    const tokenId = Number(entry);
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      throw new Error(`${label}[${index}] must be an integer in [0, ${vocabSize}), got "${String(entry)}".`);
    }
    return tokenId;
  });
  if (tokenIds.length === 0) {
    throw new Error(`${label} must not be empty.`);
  }
  return tokenIds;
}

export async function traceActivationHealth(label, buffer, dtype, elementCount, representation = undefined) {
  if (!isTraceEnabled('logits') || !isGpuBufferInstance(buffer)) {
    return;
  }
  const bytesPerElement = tensorElementBytes(dtype, representation);
  const data = await readBuffer(buffer, elementCount * bytesPerElement);
  trace.logits(label, getLogitsHealth(decodeReadback(data, dtype, representation)));
}

export function ownsBorrowedWeightBuffer(weight) {
  return !isGpuBufferInstance(weight) && !isWeightBuffer(weight);
}

export function borrowLinearWeight(weight, label) {
  if (!weight) {
    throw new Error(`DiffusionGemma self-conditioning missing ${label}.`);
  }
  if (isSplitWeightBuffer(weight)) {
    throw new Error(
      `DiffusionGemma self-conditioning does not support split weight storage for ${label}.`
    );
  }
  return {
    value: getPipelineWeightBuffer(weight, label),
    owned: ownsBorrowedWeightBuffer(weight),
  };
}

export function borrowNormWeight(weight, label) {
  if (!weight) {
    throw new Error(`DiffusionGemma self-conditioning missing ${label}.`);
  }
  if (isSplitWeightBuffer(weight)) {
    throw new Error(
      `DiffusionGemma self-conditioning does not support split norm storage for ${label}.`
    );
  }
  return {
    value: getPipelineNormWeightBuffer(weight, label),
    owned: ownsBorrowedWeightBuffer(weight),
  };
}

export function releaseBorrowedWeight(borrowed) {
  if (!borrowed?.owned) {
    return;
  }
  const value = borrowed.value;
  releaseBuffer(isWeightBuffer(value) ? value.buffer : value);
}

export function canUseChunkedSoftEmbeddingLogits(logitsState, embeddingWeight, embeddingTranspose) {
  return logitsState != null
    && isWeightBuffer(embeddingWeight)
    && getWeightDtype(embeddingWeight) === 'f16'
    && getLayout(embeddingWeight) === 'row'
    && embeddingTranspose !== true;
}

export function resolveDiffusionGemmaSoftEmbeddingChunkRows(runtimeConfig) {
  const value = runtimeConfig?.inference?.diffusionGemma?.softEmbeddingLogitsChunkRows;
  if (value == null) {
    return undefined;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(
      'runtime.inference.diffusionGemma.softEmbeddingLogitsChunkRows must be a positive integer.'
    );
  }
  return value;
}

export function normalizeCanvasTokenIds(canvas, context) {
  if (!Array.isArray(canvas) && !ArrayBuffer.isView(canvas)) {
    throw new Error(`[DiffusionGemma] ${context}.canvas must be an array or typed array of token IDs.`);
  }
  return Array.from(canvas, (value, index) => {
    if (!Number.isFinite(value) || Math.floor(value) !== value || value < 0) {
      throw new Error(`[DiffusionGemma] ${context}.canvas[${index}] must be a non-negative integer token ID.`);
    }
    return value;
  });
}

export function normalizeSelfConditioningLogits(logits, canvasLength, vocabSize) {
  if (logits == null) {
    return null;
  }
  const expected = canvasLength * vocabSize;
  const values = logits instanceof Float32Array
    ? logits
    : (Array.isArray(logits) ? Float32Array.from(logits) : null);
  if (!values) {
    throw new Error('[DiffusionGemma] selfConditioningLogits must be a Float32Array or number array.');
  }
  if (values.length !== expected) {
    throw new Error(
      `[DiffusionGemma] selfConditioningLogits length mismatch: expected ${expected}, got ${values.length}.`
    );
  }
  return values;
}

export function normalizeSelfConditioningLogitsState(logits, canvasLength, vocabSize) {
  if (logits == null || !isGpuBufferInstance(logits.logitsBuffer)) {
    return null;
  }
  const dtype = logits.logitsDtype;
  if (dtype !== 'f32') {
    throw new Error(`[DiffusionGemma] GPU selfConditioningLogits require f32 logits, got "${dtype}".`);
  }
  if (logits.vocabSize !== vocabSize) {
    throw new Error(
      `[DiffusionGemma] GPU selfConditioningLogits vocab mismatch: expected ${vocabSize}, got ${logits.vocabSize}.`
    );
  }
  if (logits.canvasLength !== canvasLength) {
    throw new Error(
      `[DiffusionGemma] GPU selfConditioningLogits canvas mismatch: expected ${canvasLength}, got ${logits.canvasLength}.`
    );
  }
  const temperature = logits.temperature ?? 1.0;
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error('[DiffusionGemma] GPU selfConditioningLogits temperature must be positive.');
  }
  return {
    logitsBuffer: logits.logitsBuffer,
    logitsDtype: dtype,
    temperature,
    releaseOnUse: logits.releaseOnUse !== false,
  };
}

export function normalizeSelfConditioningSoftEmbeddingState(state, canvasLength, hiddenSize) {
  if (state?.kind !== 'soft_embedding') {
    return null;
  }
  if (!isGpuBufferInstance(state.buffer)) {
    throw new Error('[DiffusionGemma] GPU selfConditioning soft embedding requires a GPU buffer.');
  }
  const dtype = state.dtype ?? 'f32';
  if (dtype !== 'f32') {
    throw new Error(`[DiffusionGemma] GPU selfConditioning soft embedding requires f32 dtype, got "${dtype}".`);
  }
  if (state.canvasLength !== canvasLength) {
    throw new Error(
      `[DiffusionGemma] GPU selfConditioning soft embedding canvas mismatch: ` +
      `expected ${canvasLength}, got ${state.canvasLength}.`
    );
  }
  if (state.hiddenSize !== hiddenSize) {
    throw new Error(
      `[DiffusionGemma] GPU selfConditioning soft embedding hidden mismatch: ` +
      `expected ${hiddenSize}, got ${state.hiddenSize}.`
    );
  }
  return {
    buffer: state.buffer,
    dtype,
    releaseOnUse: state.releaseOnUse !== false,
    scaled: state.scaled === true,
  };
}

export let intentBundleModulePromise = null;

export async function getExperimentalIntentBundleModule() {
  intentBundleModulePromise ??= import('../../../../experimental/hotswap/intent-bundle.js');
  return intentBundleModulePromise;
}
