import {
  createWeightBuffer,
  getBufferDtype,
  getLayout,
  getWeightDtype,
  isCpuWeightBuffer,
  isGpuBufferInstance,
  isWeightBuffer,
} from '../../../gpu/weight-buffer.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  castF16ToF32,
  castF32ToF16,
  recordActivationStaticQdq,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordScale,
  runActivationStaticQdq,
  runScale,
} from '../../../gpu/kernel-selector.js';
import { getNormWeightBuffer, getWeightBuffer } from './weights.js';
import { doCast, doMatmul, doRMSNorm, releaseOrTrack } from './ops.js';
import { runProbes } from './probes.js';
import { embed, isRangeBackedCpuEmbeddingSource, normalizeRangeBytes, decodeRangeChunkIntoOutput } from './embed.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';

const pleRuntimeCache = new WeakMap();
const pleRangeRowCache = new WeakMap();
const PLE_ACTIVATION_STATIC_QMIN = -128;
const PLE_ACTIVATION_STATIC_QMAX = 127;

function getPerLayerInputWeights(context) {
  const weights = context.weights.get('per_layer_inputs');
  if (!weights || typeof weights !== 'object') {
    throw new Error(
      'Gemma 4 per-layer inputs require global per-layer input weights, ' +
      'but state.weights.get("per_layer_inputs") was missing.'
    );
  }
  return weights;
}

function normalizePleProjectionNormDtype(dtype) {
  if (typeof dtype !== 'string') {
    return null;
  }
  const value = dtype.toLowerCase();
  if (value === 'f16' || value === 'f32') {
    return value;
  }
  return null;
}

function getPleProjectionNormDtype(weight) {
  return normalizePleProjectionNormDtype(getWeightDtype(weight))
    ?? normalizePleProjectionNormDtype(getBufferDtype(weight))
    ?? null;
}

function isRangeBackedPleProjectionSource(value) {
  return isRangeBackedCpuEmbeddingSource(value);
}

function getPleProjectionWeightDtype(weight) {
  if (isCpuWeightBuffer(weight)) {
    return weight.dtype ?? null;
  }
  return getWeightDtype(weight);
}

function isDensePleProjectionDtype(dtype) {
  return dtype === 'f16' || dtype === 'f32';
}

function resolvePleActivationStaticScaleValue(value, label) {
  if (value == null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(
      `Gemma 4 per-layer projection activation static scale "${label}" must be > 0. ` +
      `Got "${String(value)}".`
    );
  }
  return parsed;
}

function resolvePleActivationStaticQdqConfig(perLayerInputWeights) {
  const inputScale = resolvePleActivationStaticScaleValue(
    perLayerInputWeights?.perLayerModelProjectionInputActivationStaticScale,
    'input'
  );
  const outputScale = resolvePleActivationStaticScaleValue(
    perLayerInputWeights?.perLayerModelProjectionOutputActivationStaticScale,
    'output'
  );
  if (inputScale == null && outputScale == null) {
    return null;
  }
  if (inputScale == null || outputScale == null) {
    throw new Error(
      'Gemma 4 per-layer projection static activation quantization requires both input and output scales.'
    );
  }
  return {
    inputScale,
    outputScale,
  };
}

function getTensorElementCount(tensor, label) {
  if (!Array.isArray(tensor?.shape) || tensor.shape.length === 0) {
    throw new Error(`${label} requires a tensor with an explicit shape.`);
  }
  const count = tensor.shape.reduce((total, dimension) => total * Number(dimension), 1);
  if (!Number.isFinite(count) || count <= 0) {
    throw new Error(`${label} resolved an invalid tensor element count (${String(count)}).`);
  }
  return count;
}

async function applyPleActivationStaticQdq(tensor, scale, recorder, decodeBuffers, label) {
  const count = getTensorElementCount(tensor, label);
  if (tensor.dtype === 'f32') {
    return recorder
      ? await recordActivationStaticQdq(recorder, tensor, scale, {
        count,
        qmin: PLE_ACTIVATION_STATIC_QMIN,
        qmax: PLE_ACTIVATION_STATIC_QMAX,
      })
      : await runActivationStaticQdq(tensor, scale, {
        count,
        qmin: PLE_ACTIVATION_STATIC_QMIN,
        qmax: PLE_ACTIVATION_STATIC_QMAX,
      });
  }
  if (tensor.dtype !== 'f16') {
    throw new Error(
      `${label} requires f16/f32 activations for static activation quantize/dequantize. ` +
      `Got "${String(tensor.dtype)}".`
    );
  }

  let widenedTensor = null;
  let qdqTensor = null;
  try {
    widenedTensor = recorder
      ? await recordCastF16ToF32(recorder, tensor)
      : await castF16ToF32(tensor);
    qdqTensor = recorder
      ? await recordActivationStaticQdq(recorder, widenedTensor, scale, {
        count,
        qmin: PLE_ACTIVATION_STATIC_QMIN,
        qmax: PLE_ACTIVATION_STATIC_QMAX,
      })
      : await runActivationStaticQdq(widenedTensor, scale, {
        count,
        qmin: PLE_ACTIVATION_STATIC_QMIN,
        qmax: PLE_ACTIVATION_STATIC_QMAX,
      });
    return recorder
      ? await recordCastF32ToF16(recorder, qdqTensor)
      : await castF32ToF16(qdqTensor);
  } finally {
    if (qdqTensor) {
      releaseOrTrack(recorder, qdqTensor.buffer, decodeBuffers);
    }
    if (widenedTensor) {
      releaseOrTrack(recorder, widenedTensor.buffer, decodeBuffers);
    }
  }
}

export function resolveDensePleProjectionWeight(
  weight,
  label = 'per_layer_model_projection'
) {
  if (isWeightBuffer(weight)) {
    const dtype = String(weight.dtype ?? '').toLowerCase();
    if (!isDensePleProjectionDtype(dtype)) {
      throw new Error(
        'Gemma 4 sliced per-layer projection requires a dense f16/f32 base materialization. ' +
        `Got dtype="${String(weight.dtype)}" for "${label}".`
      );
    }
    if (!weight.materializations?.q4k?.buffer) {
      return weight;
    }
    return createWeightBuffer(
      weight.buffer,
      dtype,
      weight.layout,
      [...weight.shape],
      weight.label ?? label
    );
  }

  if (isGpuBufferInstance(weight)) {
    const dtype = String(getBufferDtype(weight) ?? '').toLowerCase();
    if (!isDensePleProjectionDtype(dtype)) {
      throw new Error(
        'Gemma 4 sliced per-layer projection requires dense f16/f32 GPU weights with runtime dtype metadata. ' +
        `Got dtype="${String(getBufferDtype(weight) ?? 'unknown')}" for "${label}".`
      );
    }
  }

  return weight;
}

export async function loadRangeBackedPleProjectionSliceBytes(
  weight,
  layerIdx,
  hiddenSizePerLayerInput,
  hiddenSize,
  label = 'Range-backed PLE projection slice'
) {
  if (!isCpuWeightBuffer(weight)) {
    return null;
  }
  const cpuData = weight.data;
  if (!isRangeBackedPleProjectionSource(cpuData)) {
    return null;
  }

  const dtype = getPleProjectionWeightDtype(weight);
  const layout = getLayout(weight) ?? weight.layout ?? 'row';
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', {
    dtype,
  });
  const hiddenOffset = layerIdx * hiddenSizePerLayerInput;
  const byteOffset = hiddenOffset * hiddenSize * bytesPerElement;
  const byteLength = hiddenSizePerLayerInput * hiddenSize * bytesPerElement;
  const bytes = normalizeRangeBytes(
    await cpuData.loadRange(byteOffset, byteLength),
    label
  );
  if (bytes.byteLength !== byteLength) {
    throw new Error(
      `${label} short read for layer ${layerIdx}: expected ${byteLength} bytes, got ${bytes.byteLength}.`
    );
  }
  return {
    bytes,
    dtype,
    layout,
    shape: [hiddenSizePerLayerInput, hiddenSize],
  };
}

export function inferPleProjectionNormDtype(weight, hiddenSizePerLayerInput) {
  const explicitDtype = getPleProjectionNormDtype(weight);
  if (explicitDtype) {
    return explicitDtype;
  }

  if (!isGpuBufferInstance(weight)) {
    return 'f32';
  }

  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    throw new Error('Gemma 4 per-layer projection norm dtype inference requires hiddenSizePerLayerInput > 0.');
  }

  const bytesPerElement = weight.size / hiddenSizePerLayerInput;
  if (bytesPerElement !== 2 && bytesPerElement !== 4) {
    throw new Error(
      'Gemma 4 per-layer projection norm buffer has unsupported byte size: ' +
      `bufferSize=${weight.size}, hiddenSizePerLayerInput=${hiddenSizePerLayerInput}.`
    );
  }
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });
}

function createPleProjectionNormTensor(buffer, dtype, hiddenSizePerLayerInput, label) {
  return createTensor(buffer, dtype, [hiddenSizePerLayerInput], label);
}

function destroyPleRuntimeCacheEntry(entry) {
  const cachedBuffer = entry?.scaledProjectionNormWeight?.buffer ?? null;
  if (cachedBuffer) {
    releaseBuffer(cachedBuffer);
  }
}

function destroyPleSplitTables(perLayerInputWeights) {
  const splitTables = Array.isArray(perLayerInputWeights?.embedTokensPerLayerSplit)
    ? perLayerInputWeights.embedTokensPerLayerSplit
    : null;
  if (!splitTables) {
    return;
  }
  for (const table of splitTables) {
    const buffer = table?.buffer ?? table ?? null;
    if (buffer) {
      releaseBuffer(buffer);
    }
  }
  delete perLayerInputWeights.embedTokensPerLayerSplit;
}

function destroyPleHotVocabularyRuntime(perLayerInputWeights) {
  const runtime = perLayerInputWeights?.embedTokensPerLayerHotRuntime ?? null;
  if (!runtime || typeof runtime !== 'object') {
    return;
  }
  const splitTables = Array.isArray(runtime.splitTables) ? runtime.splitTables : [];
  for (const table of splitTables) {
    const buffer = table?.buffer ?? table ?? null;
    if (buffer) {
      releaseBuffer(buffer);
    }
  }
  const mapBuffer = runtime.hotTokenIndexMapBuffer ?? null;
  if (mapBuffer) {
    releaseBuffer(mapBuffer);
  }
  delete perLayerInputWeights.embedTokensPerLayerHotRuntime;
}

export function destroyPleRuntimeCache(perLayerInputWeights) {
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return;
  }
  const entry = pleRuntimeCache.get(perLayerInputWeights);
  if (entry) {
    destroyPleRuntimeCacheEntry(entry);
    pleRuntimeCache.delete(perLayerInputWeights);
  }
  destroyPleSplitTables(perLayerInputWeights);
  destroyPleHotVocabularyRuntime(perLayerInputWeights);
  if (perLayerInputWeights.embedTokensPerLayer) {
    pleRangeRowCache.delete(perLayerInputWeights.embedTokensPerLayer);
  }
}

export function scalePerLayerProjectionNormWeights(weight, combineScale, rmsNormWeightOffset = false) {
  if (rmsNormWeightOffset) {
    return null;
  }
  const source = isCpuWeightBuffer(weight) ? weight.data : weight;
  const isArrayLikeView = ArrayBuffer.isView(source) && typeof source.length === 'number';
  if (!(source instanceof Float32Array) && !isArrayLikeView && !Array.isArray(source)) {
    return null;
  }
  const scaled = Float32Array.from(source);
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] *= combineScale;
  }
  return scaled;
}

export async function ensurePleScaledProjectionNormWeight(context, combineScale = 2 ** -0.5) {
  const hiddenSizePerLayerInput = Number(context?.config?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }
  if (context?.weightConfig?.rmsNormWeightOffset) {
    return null;
  }

  const perLayerInputWeights = getPerLayerInputWeights(context);
  const cachedEntry = pleRuntimeCache.get(perLayerInputWeights) ?? null;
  if (cachedEntry?.combineScale === combineScale && cachedEntry.scaledProjectionNormWeight) {
    return cachedEntry.scaledProjectionNormWeight;
  }
  if (cachedEntry) {
    destroyPleRuntimeCacheEntry(cachedEntry);
    pleRuntimeCache.delete(perLayerInputWeights);
  }

  const projectionNormWeight = perLayerInputWeights.perLayerProjectionNorm;
  if (!projectionNormWeight) {
    return null;
  }

  let scaledProjectionNormWeight = null;
  if (isGpuBufferInstance(projectionNormWeight)) {
    const projectionNormDtype = inferPleProjectionNormDtype(
      projectionNormWeight,
      hiddenSizePerLayerInput
    );
    const scaledTensor = await runScale(
      createPleProjectionNormTensor(
        projectionNormWeight,
        projectionNormDtype,
        hiddenSizePerLayerInput,
        'per_layer_projection_norm'
      ),
      combineScale,
      { count: hiddenSizePerLayerInput }
    );
    scaledProjectionNormWeight = createPleProjectionNormTensor(
      scaledTensor.buffer,
      projectionNormDtype,
      hiddenSizePerLayerInput,
      'per_layer_projection_norm_scaled'
    );
  } else {
    const scaledValues = scalePerLayerProjectionNormWeights(
      projectionNormWeight,
      combineScale,
      false
    );
    if (!scaledValues) {
      return null;
    }
    if (scaledValues.length !== hiddenSizePerLayerInput) {
      throw new Error(
        'Gemma 4 per-layer projection norm cache shape mismatch: ' +
        `expected ${hiddenSizePerLayerInput} values, got ${scaledValues.length}.`
      );
    }
    const device = getDevice();
    if (!device) {
      throw new Error('No GPU device available for Gemma 4 per-layer projection norm cache.');
    }
    const scaledBuffer = acquireBuffer(
      scaledValues.byteLength,
      undefined,
      'per_layer_projection_norm_scaled'
    );
    try {
      device.queue.writeBuffer(scaledBuffer, 0, scaledValues);
    } catch (error) {
      releaseBuffer(scaledBuffer);
      throw error;
    }
    scaledProjectionNormWeight = createPleProjectionNormTensor(
      scaledBuffer,
      getPleProjectionNormDtype(projectionNormWeight) ?? 'f32',
      hiddenSizePerLayerInput,
      'per_layer_projection_norm_scaled'
    );
  }

  if (scaledProjectionNormWeight?.shape?.[0] !== hiddenSizePerLayerInput) {
    throw new Error(
      'Gemma 4 per-layer projection norm cache shape mismatch after tensor creation: ' +
      `expected ${hiddenSizePerLayerInput}, got ${scaledProjectionNormWeight?.shape?.[0] ?? 'unknown'}.`
    );
  }
  if (!normalizePleProjectionNormDtype(scaledProjectionNormWeight?.dtype)) {
    throw new Error(
      'Gemma 4 per-layer projection norm cache produced an invalid dtype: ' +
      `"${String(scaledProjectionNormWeight?.dtype ?? 'undefined')}".`
    );
  }

  pleRuntimeCache.set(perLayerInputWeights, {
    combineScale,
    scaledProjectionNormWeight,
  });
  return scaledProjectionNormWeight;
}

function getEmbeddingSource(weight, label) {
  if (isWeightBuffer(weight)) {
    return weight.buffer;
  }
  if (isCpuWeightBuffer(weight) || isGpuBufferInstance(weight) || weight instanceof Float32Array) {
    return weight;
  }
  throw new Error(`Gemma 4 per-layer input ${label} has unsupported type "${weight?.constructor?.name ?? typeof weight}".`);
}

function getEmbeddingDtype(weight) {
  if (isCpuWeightBuffer(weight)) {
    return weight.dtype;
  }
  return getWeightDtype(weight);
}

function getEmbeddingTranspose(weight) {
  if (isWeightBuffer(weight) || isCpuWeightBuffer(weight)) {
    return weight.layout === 'column';
  }
  return false;
}

// Step 4: Pre-allocated buffer cache for decode-path fused projection slices.
// Avoids per-step acquireBuffer/releaseBuffer churn for the 35 slice buffers.
export function createPleBufferCache(numLayers, sliceBytes) {
  const sliceBuffers = Array.from({ length: numLayers }, (_, l) =>
    acquireBuffer(sliceBytes, undefined, `L${l}.ple_slice_cached`));
  const gatherSliceBuffers = Array.from({ length: numLayers }, (_, l) =>
    acquireBuffer(sliceBytes, undefined, `L${l}.ple_gather_slice_cached`));
  const ownedBuffers = new Set();
  for (const buffer of sliceBuffers) {
    if (buffer) ownedBuffers.add(buffer);
  }
  for (const buffer of gatherSliceBuffers) {
    if (buffer) ownedBuffers.add(buffer);
  }
  return {
    sliceBuffers,
    gatherSliceBuffers,
    preparedTokenEntries: new Map(),
    preparedTokenBytes: 0,
    ownedBuffers,
  };
}

export function destroyPleBufferCache(cache) {
  if (!cache?.sliceBuffers && !cache?.gatherSliceBuffers) return;
  for (const buf of cache?.sliceBuffers ?? []) {
    if (buf) releaseBuffer(buf);
  }
  for (const buf of cache?.gatherSliceBuffers ?? []) {
    if (buf) releaseBuffer(buf);
  }
  for (const entry of cache?.preparedTokenEntries?.values?.() ?? []) {
    for (const buf of entry?.buffers ?? []) {
      if (buf) releaseBuffer(buf);
    }
  }
  cache.sliceBuffers = null;
  cache.gatherSliceBuffers = null;
  cache.preparedTokenEntries = null;
  cache.preparedTokenBytes = 0;
  cache.ownedBuffers = null;
}

function isCachedPleSliceBuffer(cache, buffer) {
  return cache?.ownedBuffers instanceof Set && cache.ownedBuffers.has(buffer);
}

function releasePleSliceBuffer(recorder, buffer, decodeBuffers, cache) {
  if (!buffer || isCachedPleSliceBuffer(cache, buffer)) {
    return;
  }
  releaseOrTrack(recorder, buffer, decodeBuffers);
}

function getPleRowCachePolicy(sessionConfig) {
  const rowCache = sessionConfig?.rowCache ?? null;
  if (!rowCache || rowCache.mode === 'off') {
    return null;
  }
  if (rowCache.mode !== 'lru') {
    throw new Error(
      `Gemma 4 per-layer input row cache mode "${String(rowCache.mode)}" is not implemented.`
    );
  }
  const decodedDtype = String(rowCache.decodedDtype ?? '').toLowerCase();
  if (decodedDtype !== 'f32') {
    throw new Error(
      `Gemma 4 range-backed per-layer input row cache requires rowCache.decodedDtype="f32"; ` +
      `got "${String(rowCache.decodedDtype)}".`
    );
  }
  const maxRows = Math.trunc(Number(rowCache.maxRows));
  const maxBytes = Math.trunc(Number(rowCache.maxBytes));
  if (!Number.isFinite(maxRows) || maxRows <= 0) {
    throw new Error('Gemma 4 per-layer input row cache requires rowCache.maxRows > 0.');
  }
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error('Gemma 4 per-layer input row cache requires rowCache.maxBytes > 0.');
  }
  return { maxRows, maxBytes };
}

function getPleHotCachePolicy(sessionConfig) {
  const hotCache = sessionConfig?.hotCache ?? null;
  if (!hotCache || hotCache.mode === 'off') {
    return null;
  }
  if (hotCache.mode === 'tokenizer_scores') {
    const outputDtype = String(hotCache.outputDtype ?? '').toLowerCase();
    if (outputDtype !== 'f16' && outputDtype !== 'f32') {
      throw new Error(
        `Gemma 4 per-layer input hot vocabulary cache requires hotCache.outputDtype to be "f16" or "f32"; ` +
        `got "${String(hotCache.outputDtype)}".`
      );
    }
    const maxTokens = Math.trunc(Number(hotCache.maxTokens));
    const maxBytes = Math.trunc(Number(hotCache.maxBytes));
    if (!Number.isFinite(maxTokens) || maxTokens <= 0) {
      throw new Error('Gemma 4 per-layer input hot vocabulary cache requires hotCache.maxTokens > 0.');
    }
    if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
      throw new Error('Gemma 4 per-layer input hot vocabulary cache requires hotCache.maxBytes > 0.');
    }
    return {
      mode: 'tokenizer_scores',
      maxTokens,
      maxBytes,
      outputDtype,
    };
  }
  if (hotCache.mode !== 'prepared_tokens') {
    throw new Error(
      `Gemma 4 per-layer input hot cache mode "${String(hotCache.mode)}" is not implemented.`
    );
  }
  const outputDtype = String(hotCache.outputDtype ?? '').toLowerCase();
  if (outputDtype !== 'f16' && outputDtype !== 'f32') {
    throw new Error(
      `Gemma 4 per-layer input hot cache requires hotCache.outputDtype to be "f16" or "f32"; ` +
      `got "${String(hotCache.outputDtype)}".`
    );
  }
  const maxTokens = Math.trunc(Number(hotCache.maxTokens));
  const maxBytes = Math.trunc(Number(hotCache.maxBytes));
  if (!Number.isFinite(maxTokens) || maxTokens <= 0) {
    throw new Error('Gemma 4 per-layer input hot cache requires hotCache.maxTokens > 0.');
  }
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error('Gemma 4 per-layer input hot cache requires hotCache.maxBytes > 0.');
  }
  return { mode: 'prepared_tokens', maxTokens, maxBytes, outputDtype };
}

export function getPleHotVocabularyRuntime(context) {
  const perLayerInputWeights = context?.weights?.get?.('per_layer_inputs');
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return null;
  }
  const runtime = perLayerInputWeights.embedTokensPerLayerHotRuntime ?? null;
  return runtime && typeof runtime === 'object' ? runtime : null;
}

function getPleSplitTablePolicy(sessionConfig) {
  if (sessionConfig?.materialization !== 'gpu_split_tables') {
    return null;
  }
  return { mode: 'gpu_split_tables' };
}

function releasePreparedTokenEntry(cache, tokenId, entry) {
  if (!cache || !entry) {
    return;
  }
  cache.preparedTokenEntries?.delete(tokenId);
  cache.preparedTokenBytes -= entry?.bytes ?? 0;
  for (const buffer of entry.buffers ?? []) {
    if (!buffer) continue;
    cache.ownedBuffers?.delete(buffer);
    releaseBuffer(buffer);
  }
}

function prunePreparedTokenCache(cache, policy) {
  while (
    cache.preparedTokenEntries.size > policy.maxTokens
    || cache.preparedTokenBytes > policy.maxBytes
  ) {
    const oldest = cache.preparedTokenEntries.keys().next();
    if (oldest.done) {
      break;
    }
    const tokenId = oldest.value;
    const entry = cache.preparedTokenEntries.get(tokenId);
    releasePreparedTokenEntry(cache, tokenId, entry);
  }
}

function getPreparedTokenEntry(cache, tokenId, sessionConfig, activationDtype, stats = null) {
  const policy = getPleHotCachePolicy(sessionConfig);
  if (!policy || policy.mode !== 'prepared_tokens' || !(cache?.preparedTokenEntries instanceof Map)) {
    return null;
  }
  if (policy.outputDtype !== activationDtype) {
    throw new Error(
      `Gemma 4 prepared per-layer input hot cache requires activation dtype "${policy.outputDtype}", ` +
      `got "${String(activationDtype)}".`
    );
  }
  const entry = cache.preparedTokenEntries.get(tokenId) ?? null;
  if (!entry) {
    if (stats) {
      stats.plePreparedTokenCacheMisses = (stats.plePreparedTokenCacheMisses ?? 0) + 1;
      stats.plePreparedTokenCacheEntries = cache.preparedTokenEntries.size;
      stats.plePreparedTokenCacheBytes = cache.preparedTokenBytes;
    }
    return null;
  }
  cache.preparedTokenEntries.delete(tokenId);
  cache.preparedTokenEntries.set(tokenId, entry);
  if (stats) {
    stats.plePreparedTokenCacheHits = (stats.plePreparedTokenCacheHits ?? 0) + 1;
    stats.plePreparedTokenCacheEntries = cache.preparedTokenEntries.size;
    stats.plePreparedTokenCacheBytes = cache.preparedTokenBytes;
  }
  return entry.buffers.slice();
}

function storePreparedTokenEntry(cache, tokenId, buffers, sessionConfig, activationDtype, stats = null) {
  const policy = getPleHotCachePolicy(sessionConfig);
  if (!policy || policy.mode !== 'prepared_tokens' || !(cache?.preparedTokenEntries instanceof Map)) {
    return buffers;
  }
  if (policy.outputDtype !== activationDtype) {
    throw new Error(
      `Gemma 4 prepared per-layer input hot cache requires activation dtype "${policy.outputDtype}", ` +
      `got "${String(activationDtype)}".`
    );
  }
  const existing = cache.preparedTokenEntries.get(tokenId) ?? null;
  if (existing) {
    releasePreparedTokenEntry(cache, tokenId, existing);
  }
  const cachedBuffers = buffers.slice();
  const bytes = cachedBuffers.reduce((total, buffer) => total + (buffer?.size ?? 0), 0);
  for (const buffer of cachedBuffers) {
    if (buffer) {
      cache.ownedBuffers?.add(buffer);
    }
  }
  cache.preparedTokenEntries.set(tokenId, { buffers: cachedBuffers, bytes });
  cache.preparedTokenBytes += bytes;
  prunePreparedTokenCache(cache, policy);
  if (stats) {
    stats.plePreparedTokenCacheEntries = cache.preparedTokenEntries.size;
    stats.plePreparedTokenCacheBytes = cache.preparedTokenBytes;
  }
  return cachedBuffers.slice();
}

function getPleRangeRowLoadConfig(embedTokensPerLayer, totalPerLayerHiddenSize) {
  const sourceDtype = String(
    (isCpuWeightBuffer(embedTokensPerLayer) ? embedTokensPerLayer.data?.sourceDtype : null)
      ?? embedTokensPerLayer?.dtype
      ?? 'f32'
  ).toLowerCase();
  const bytesPerElement = (sourceDtype === 'f16' || sourceDtype === 'bf16') ? 2 : 4;
  return {
    sourceDtype,
    sourceRowBytes: totalPerLayerHiddenSize * bytesPerElement,
  };
}

function getPleRangeRowCache(embedTokensPerLayer, sessionConfig) {
  const policy = getPleRowCachePolicy(sessionConfig);
  if (!policy) {
    return null;
  }
  const cached = pleRangeRowCache.get(embedTokensPerLayer);
  if (cached && cached.maxRows === policy.maxRows && cached.maxBytes === policy.maxBytes) {
    return cached;
  }
  const next = {
    maxRows: policy.maxRows,
    maxBytes: policy.maxBytes,
    totalBytes: 0,
    rows: new Map(),
  };
  pleRangeRowCache.set(embedTokensPerLayer, next);
  return next;
}

function touchPleCachedRow(cache, tokenId) {
  const hit = cache?.rows?.get(tokenId) ?? null;
  if (!hit) {
    return null;
  }
  cache.rows.delete(tokenId);
  cache.rows.set(tokenId, hit);
  return hit.row;
}

function prunePleRangeRowCache(cache) {
  while (cache.rows.size > cache.maxRows || cache.totalBytes > cache.maxBytes) {
    const oldest = cache.rows.keys().next();
    if (oldest.done) {
      break;
    }
    const entry = cache.rows.get(oldest.value);
    cache.rows.delete(oldest.value);
    cache.totalBytes -= entry?.bytes ?? 0;
  }
}

function cachePleRangeRow(cache, tokenId, row) {
  if (!cache) {
    return row;
  }
  const existing = cache.rows.get(tokenId);
  if (existing) {
    cache.totalBytes -= existing.bytes;
    cache.rows.delete(tokenId);
  }
  cache.rows.set(tokenId, { row, bytes: row.byteLength });
  cache.totalBytes += row.byteLength;
  prunePleRangeRowCache(cache);
  return row;
}

async function loadRangeBackedPleRow(
  tokenId,
  embedTokensPerLayer,
  totalPerLayerHiddenSize,
  sessionConfig,
  label,
  prefetchedRow = null
) {
  if (!isCpuWeightBuffer(embedTokensPerLayer)) {
    return null;
  }
  const cpuData = embedTokensPerLayer.data;
  if (!isRangeBackedCpuEmbeddingSource(cpuData)) {
    return null;
  }

  const cache = getPleRangeRowCache(embedTokensPerLayer, sessionConfig);
  if (prefetchedRow && prefetchedRow.tokenId === tokenId) {
    return cachePleRangeRow(cache, tokenId, prefetchedRow.row);
  }

  const cached = touchPleCachedRow(cache, tokenId);
  if (cached) {
    return cached;
  }

  const { sourceDtype, sourceRowBytes } = getPleRangeRowLoadConfig(
    embedTokensPerLayer,
    totalPerLayerHiddenSize
  );
  const chunk = normalizeRangeBytes(
    await cpuData.loadRange(tokenId * sourceRowBytes, sourceRowBytes),
    label
  );
  const row = new Float32Array(totalPerLayerHiddenSize);
  decodeRangeChunkIntoOutput(chunk, sourceDtype, row, 0, totalPerLayerHiddenSize);
  return cachePleRangeRow(cache, tokenId, row);
}

// Step 5: Prefetch next token's PLE row during current decode step.
// Returns a promise resolving to { tokenId, row: Float32Array } or null.
// Call after sampling produces the next token; pass result as options.prefetchedRow
// to the next preparePerLayerInputs call.
export function prefetchPerLayerRow(tokenId, embedTokensPerLayer, totalPerLayerHiddenSize, sessionConfig = null) {
  if (!isCpuWeightBuffer(embedTokensPerLayer)) return null;
  const cpuData = embedTokensPerLayer.data;
  if (!isRangeBackedCpuEmbeddingSource(cpuData)) return null;
  return loadRangeBackedPleRow(
    tokenId,
    embedTokensPerLayer,
    totalPerLayerHiddenSize,
    sessionConfig,
    'Prefetched PLE row'
  )
    .then(row => (row ? { tokenId, row } : null))
    .catch(() => null);
}

export function hasRangeBackedPerLayerInputEmbeddings(context) {
  const hiddenSizePerLayerInput = Number(context?.config?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return false;
  }

  const perLayerInputWeights = context?.weights?.get?.('per_layer_inputs');
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return false;
  }
  if (Array.isArray(perLayerInputWeights.embedTokensPerLayerSplit) && perLayerInputWeights.embedTokensPerLayerSplit.length > 0) {
    return false;
  }

  const embedTokensPerLayer = perLayerInputWeights.embedTokensPerLayer;
  return isCpuWeightBuffer(embedTokensPerLayer)
    && isRangeBackedCpuEmbeddingSource(embedTokensPerLayer.data);
}

export function hasGpuSplitPerLayerInputEmbeddings(context) {
  const hiddenSizePerLayerInput = Number(context?.config?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return false;
  }

  const perLayerInputWeights = context?.weights?.get?.('per_layer_inputs');
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return false;
  }

  return Array.isArray(perLayerInputWeights.embedTokensPerLayerSplit)
    && perLayerInputWeights.embedTokensPerLayerSplit.length > 0;
}

export async function ensurePleGpuSplitTablesRuntime(context) {
  const policy = getPleSplitTablePolicy(context?.perLayerInputsSession ?? null);
  if (!policy) {
    return null;
  }

  const config = context?.config ?? null;
  const hiddenSizePerLayerInput = Number(config?.hiddenSizePerLayerInput ?? 0);
  const vocabSizePerLayerInput = Number(config?.vocabSizePerLayerInput ?? 0);
  const numLayers = Number(config?.numLayers ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }
  if (!Number.isFinite(vocabSizePerLayerInput) || vocabSizePerLayerInput <= 0) {
    return null;
  }
  if (!Number.isFinite(numLayers) || numLayers <= 0) {
    return null;
  }

  const perLayerInputWeights = getPerLayerInputWeights(context);
  if (Array.isArray(perLayerInputWeights.embedTokensPerLayerSplit) && perLayerInputWeights.embedTokensPerLayerSplit.length === numLayers) {
    return perLayerInputWeights.embedTokensPerLayerSplit;
  }

  const embedTokensPerLayer = perLayerInputWeights.embedTokensPerLayer;
  if (!isCpuWeightBuffer(embedTokensPerLayer) || !isRangeBackedCpuEmbeddingSource(embedTokensPerLayer.data)) {
    throw new Error('Gemma 4 gpu_split_tables materialization requires a range-backed CPU embedTokensPerLayer source.');
  }

  const sourceDtype = String(embedTokensPerLayer.data?.sourceDtype ?? embedTokensPerLayer.dtype ?? 'f32').toLowerCase();
  if (sourceDtype !== 'f16' && sourceDtype !== 'f32') {
    throw new Error(
      `Gemma 4 gpu_split_tables materialization requires f16/f32 source rows; got "${sourceDtype}".`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Gemma 4 gpu_split_tables materialization.');
  }

  const bytesPerElement = sourceDtype === 'f16' ? 2 : 4;
  const totalPerLayerHiddenSize = numLayers * hiddenSizePerLayerInput;
  const tableBytes = vocabSizePerLayerInput * hiddenSizePerLayerInput * bytesPerElement;
  const splitTables = Array.from({ length: numLayers }, (_, layerIdx) => createWeightBuffer(
    acquireBuffer(tableBytes, undefined, `L${layerIdx}.ple_table_split`),
    sourceDtype,
    'row',
    [vocabSizePerLayerInput, hiddenSizePerLayerInput],
    `L${layerIdx}.embed_tokens_per_layer_split`
  ));

  try {
    const rowsPerChunk = 128;
    for (let rowStart = 0; rowStart < vocabSizePerLayerInput; rowStart += rowsPerChunk) {
      const rowCount = Math.min(rowsPerChunk, vocabSizePerLayerInput - rowStart);
      const chunkByteOffset = rowStart * totalPerLayerHiddenSize * bytesPerElement;
      const chunkByteLength = rowCount * totalPerLayerHiddenSize * bytesPerElement;
      const chunk = normalizeRangeBytes(
        await embedTokensPerLayer.data.loadRange(chunkByteOffset, chunkByteLength),
        'Gemma 4 split GPU PLE chunk'
      );

      if (sourceDtype === 'f16') {
        const sourceWords = new Uint16Array(chunk.buffer, chunk.byteOffset, rowCount * totalPerLayerHiddenSize);
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
          const layerWords = new Uint16Array(rowCount * hiddenSizePerLayerInput);
          for (let row = 0; row < rowCount; row += 1) {
            const sourceStart = row * totalPerLayerHiddenSize + layerIdx * hiddenSizePerLayerInput;
            layerWords.set(
              sourceWords.subarray(sourceStart, sourceStart + hiddenSizePerLayerInput),
              row * hiddenSizePerLayerInput
            );
          }
          device.queue.writeBuffer(
            splitTables[layerIdx].buffer,
            rowStart * hiddenSizePerLayerInput * bytesPerElement,
            layerWords.buffer,
            layerWords.byteOffset,
            layerWords.byteLength
          );
        }
      } else {
        const sourceValues = new Float32Array(chunk.buffer, chunk.byteOffset, rowCount * totalPerLayerHiddenSize);
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
          const layerValues = new Float32Array(rowCount * hiddenSizePerLayerInput);
          for (let row = 0; row < rowCount; row += 1) {
            const sourceStart = row * totalPerLayerHiddenSize + layerIdx * hiddenSizePerLayerInput;
            layerValues.set(
              sourceValues.subarray(sourceStart, sourceStart + hiddenSizePerLayerInput),
              row * hiddenSizePerLayerInput
            );
          }
          device.queue.writeBuffer(
            splitTables[layerIdx].buffer,
            rowStart * hiddenSizePerLayerInput * bytesPerElement,
            layerValues.buffer,
            layerValues.byteOffset,
            layerValues.byteLength
          );
        }
      }
    }
  } catch (error) {
    for (const table of splitTables) {
      releaseBuffer(table.buffer);
    }
    throw error;
  }

  perLayerInputWeights.embedTokensPerLayerSplit = splitTables;
  return splitTables;
}

function resolvePleHotVocabularySeedTokenIds(context, maxTokens, vocabSizePerLayerInput) {
  const rawSeedTokenIds = Array.isArray(context?.seedTokenIds) ? context.seedTokenIds : null;
  if (!rawSeedTokenIds || rawSeedTokenIds.length === 0) {
    return [];
  }

  const specialTokenIds = new Set();
  const tokenizerSpecialTokens = context?.tokenizer?.getSpecialTokens?.() ?? null;
  if (tokenizerSpecialTokens && typeof tokenizerSpecialTokens === 'object') {
    for (const value of Object.values(tokenizerSpecialTokens)) {
      if (Number.isInteger(value)) {
        specialTokenIds.add(value);
      }
    }
  }

  const seeded = [];
  const seen = new Set();
  for (let i = rawSeedTokenIds.length - 1; i >= 0; i -= 1) {
    const tokenId = rawSeedTokenIds[i];
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSizePerLayerInput) {
      continue;
    }
    if (specialTokenIds.has(tokenId) || seen.has(tokenId)) {
      continue;
    }
    seeded.push(tokenId);
    seen.add(tokenId);
    if (seeded.length >= maxTokens) {
      break;
    }
  }
  return seeded;
}

function mergePleHotVocabularyTokenIds(seedTokenIds, tokenizerHotTokenIds, maxTokens, vocabSizePerLayerInput) {
  const merged = [];
  const seen = new Set();

  for (const tokenId of seedTokenIds ?? []) {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSizePerLayerInput || seen.has(tokenId)) {
      continue;
    }
    merged.push(tokenId);
    seen.add(tokenId);
    if (merged.length >= maxTokens) {
      return merged;
    }
  }

  for (const tokenId of tokenizerHotTokenIds ?? []) {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSizePerLayerInput || seen.has(tokenId)) {
      continue;
    }
    merged.push(tokenId);
    seen.add(tokenId);
    if (merged.length >= maxTokens) {
      break;
    }
  }

  return merged;
}

export async function ensurePleGpuHotVocabularyRuntime(context) {
  const policy = getPleHotCachePolicy(context?.perLayerInputsSession ?? null);
  if (!policy || policy.mode !== 'tokenizer_scores') {
    return null;
  }

  const config = context?.config ?? null;
  const hiddenSizePerLayerInput = Number(config?.hiddenSizePerLayerInput ?? 0);
  const vocabSizePerLayerInput = Number(config?.vocabSizePerLayerInput ?? 0);
  const numLayers = Number(config?.numLayers ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }
  if (!Number.isFinite(vocabSizePerLayerInput) || vocabSizePerLayerInput <= 0) {
    return null;
  }
  if (!Number.isFinite(numLayers) || numLayers <= 0) {
    return null;
  }

  const tokenizer = context?.tokenizer ?? null;
  const tokenizerHotTokenIds = typeof tokenizer?.getHotTokenIds === 'function'
    ? tokenizer.getHotTokenIds(policy.maxTokens)
    : null;
  const seedTokenIds = resolvePleHotVocabularySeedTokenIds(
    context,
    policy.maxTokens,
    vocabSizePerLayerInput
  );
  const hotTokenIds = mergePleHotVocabularyTokenIds(
    seedTokenIds,
    tokenizerHotTokenIds,
    policy.maxTokens,
    vocabSizePerLayerInput
  );
  if (hotTokenIds.length === 0) {
    return null;
  }
  const hotTokenIdsSignature = hotTokenIds.join(',');

  const perLayerInputWeights = getPerLayerInputWeights(context);
  const cached = perLayerInputWeights.embedTokensPerLayerHotRuntime ?? null;
  if (
    cached
    && cached.maxTokens === policy.maxTokens
    && cached.outputDtype === policy.outputDtype
    && cached.vocabSize === vocabSizePerLayerInput
    && cached.numLayers === numLayers
    && cached.hotTokenIdsSignature === hotTokenIdsSignature
  ) {
    return cached;
  }
  destroyPleHotVocabularyRuntime(perLayerInputWeights);

  const embedTokensPerLayer = perLayerInputWeights.embedTokensPerLayer;
  if (!isCpuWeightBuffer(embedTokensPerLayer) || !isRangeBackedCpuEmbeddingSource(embedTokensPerLayer.data)) {
    return null;
  }

  const sourceDtype = String(embedTokensPerLayer.data?.sourceDtype ?? embedTokensPerLayer.dtype ?? 'f32').toLowerCase();
  if (sourceDtype !== policy.outputDtype) {
    throw new Error(
      `Gemma 4 hot vocabulary cache requires source dtype "${policy.outputDtype}" for zero-copy row packing; ` +
      `got "${sourceDtype}".`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Gemma 4 hot vocabulary cache.');
  }

  const bytesPerElement = policy.outputDtype === 'f16' ? 2 : 4;
  const totalPerLayerHiddenSize = numLayers * hiddenSizePerLayerInput;
  const sentinelIndex = hotTokenIds.length;
  const hotRowCount = sentinelIndex + 1;
  const splitTableBytes = hotRowCount * hiddenSizePerLayerInput * bytesPerElement;
  const splitTables = Array.from({ length: numLayers }, (_, layerIdx) => createWeightBuffer(
    acquireBuffer(splitTableBytes, undefined, `L${layerIdx}.ple_hot_vocab_table`),
    policy.outputDtype,
    'row',
    [hotRowCount, hiddenSizePerLayerInput],
    `L${layerIdx}.embed_tokens_per_layer_hot_vocab`
  ));
  const hotTokenIndexMap = new Uint32Array(vocabSizePerLayerInput);
  hotTokenIndexMap.fill(sentinelIndex);
  for (let hotIndex = 0; hotIndex < hotTokenIds.length; hotIndex += 1) {
    const tokenId = hotTokenIds[hotIndex];
    if (Number.isInteger(tokenId) && tokenId >= 0 && tokenId < vocabSizePerLayerInput) {
      hotTokenIndexMap[tokenId] = hotIndex;
    }
  }
  const hotTokenIndexMapBuffer = acquireBuffer(
    hotTokenIndexMap.byteLength,
    undefined,
    'ple_hot_token_index_map'
  );

  try {
    device.queue.writeBuffer(hotTokenIndexMapBuffer, 0, hotTokenIndexMap);
    const zeroRow = policy.outputDtype === 'f16'
      ? new Uint16Array(hiddenSizePerLayerInput)
      : new Float32Array(hiddenSizePerLayerInput);
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
      device.queue.writeBuffer(
        splitTables[layerIdx].buffer,
        sentinelIndex * hiddenSizePerLayerInput * bytesPerElement,
        zeroRow.buffer,
        zeroRow.byteOffset,
        zeroRow.byteLength
      );
    }

    const { sourceRowBytes } = getPleRangeRowLoadConfig(embedTokensPerLayer, totalPerLayerHiddenSize);
    for (let hotIndex = 0; hotIndex < hotTokenIds.length; hotIndex += 1) {
      const tokenId = hotTokenIds[hotIndex];
      const chunk = normalizeRangeBytes(
        await embedTokensPerLayer.data.loadRange(tokenId * sourceRowBytes, sourceRowBytes),
        'Gemma 4 hot vocabulary PLE row'
      );
      if (policy.outputDtype === 'f16') {
        const sourceWords = new Uint16Array(chunk.buffer, chunk.byteOffset, totalPerLayerHiddenSize);
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
          const sourceStart = layerIdx * hiddenSizePerLayerInput;
          device.queue.writeBuffer(
            splitTables[layerIdx].buffer,
            hotIndex * hiddenSizePerLayerInput * bytesPerElement,
            sourceWords.buffer,
            sourceWords.byteOffset + sourceStart * bytesPerElement,
            hiddenSizePerLayerInput * bytesPerElement
          );
        }
      } else {
        const sourceValues = new Float32Array(chunk.buffer, chunk.byteOffset, totalPerLayerHiddenSize);
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
          const sourceStart = layerIdx * hiddenSizePerLayerInput;
          device.queue.writeBuffer(
            splitTables[layerIdx].buffer,
            hotIndex * hiddenSizePerLayerInput * bytesPerElement,
            sourceValues.buffer,
            sourceValues.byteOffset + sourceStart * bytesPerElement,
            hiddenSizePerLayerInput * bytesPerElement
          );
        }
      }
    }
  } catch (error) {
    for (const table of splitTables) {
      releaseBuffer(table.buffer);
    }
    releaseBuffer(hotTokenIndexMapBuffer);
    throw error;
  }

  const runtime = {
    mode: 'tokenizer_scores',
    maxTokens: policy.maxTokens,
    outputDtype: policy.outputDtype,
    vocabSize: vocabSizePerLayerInput,
    numLayers,
    hotTokenIdsSignature,
    hotTokenIds: Uint32Array.from(hotTokenIds),
    hotTokenIndexMap,
    hotTokenIndexMapBuffer,
    sentinelIndex,
    splitTables,
  };
  perLayerInputWeights.embedTokensPerLayerHotRuntime = runtime;
  return runtime;
}

export async function preparePerLayerInputs(tokenIds, inputEmbedsTensor, context, options = {}) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const hiddenSizePerLayerInput = Number(config.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }

  const vocabSizePerLayerInput = Number(config.vocabSizePerLayerInput ?? 0);
  if (!Number.isFinite(vocabSizePerLayerInput) || vocabSizePerLayerInput <= 0) {
    throw new Error(
      `Gemma 4 model "${config.modelId ?? 'unknown'}" requires architecture.vocabSizePerLayerInput ` +
      'when hiddenSizePerLayerInput is enabled.'
    );
  }

  const perLayerInputWeights = getPerLayerInputWeights(context);
  const embedTokensPerLayer = perLayerInputWeights.embedTokensPerLayer;
  const embedTokensPerLayerSplit = Array.isArray(perLayerInputWeights.embedTokensPerLayerSplit)
    ? perLayerInputWeights.embedTokensPerLayerSplit
    : null;
  const hotVocabularyRuntime = getPleHotVocabularyRuntime(context);
  const perLayerModelProjection = perLayerInputWeights.perLayerModelProjection;
  const perLayerProjectionNorm = perLayerInputWeights.perLayerProjectionNorm;
  if (!embedTokensPerLayer || !perLayerModelProjection || !perLayerProjectionNorm) {
    throw new Error(
      'Gemma 4 per-layer inputs require embedTokensPerLayer, perLayerModelProjection, ' +
      'and perLayerProjectionNorm weights.'
    );
  }

  const numLayers = config.numLayers;
  const hasSplitEmbeddingTables = Array.isArray(embedTokensPerLayerSplit)
    && embedTokensPerLayerSplit.length === numLayers;
  const hasHotVocabularyTables = Array.isArray(hotVocabularyRuntime?.splitTables)
    && hotVocabularyRuntime.splitTables.length === numLayers;
  const numTokens = Number.isFinite(options.numTokens) ? options.numTokens : inputEmbedsTensor.shape?.[0];
  const indexOffset = Number.isFinite(options.indexOffset) ? options.indexOffset : 0;
  const perLayerIndexOffset = Number.isFinite(options.perLayerIndexOffset)
    ? options.perLayerIndexOffset
    : indexOffset;
  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('Gemma 4 per-layer inputs require a positive numTokens value.');
  }

  const activationDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: inputEmbedsTensor.dtype,
  });
  const perLayerTokenIdsOption = options.perLayerTokenIds ?? null;
  let hotLocalTokenIds = null;
  if (
    !perLayerTokenIdsOption
    && hasHotVocabularyTables
    && numTokens === 1
    && !isGpuBufferInstance(tokenIds)
    && Array.isArray(tokenIds)
    && Number.isInteger(tokenIds[0])
  ) {
    const hotIndex = hotVocabularyRuntime.hotTokenIndexMap?.[tokenIds[0]] ?? hotVocabularyRuntime.sentinelIndex;
    if (hotIndex !== hotVocabularyRuntime.sentinelIndex) {
      hotLocalTokenIds = new Uint32Array([hotIndex]);
      if (context.stats) {
        context.stats.pleHotVocabularyHits = (context.stats.pleHotVocabularyHits ?? 0) + 1;
      }
    } else if (context.stats) {
      context.stats.pleHotVocabularyMisses = (context.stats.pleHotVocabularyMisses ?? 0) + 1;
    }
  }
  const perLayerTokenIds = perLayerTokenIdsOption ?? hotLocalTokenIds;
  const useHotVocabularyTables = hasHotVocabularyTables && perLayerTokenIds != null;

  const perLayerEmbeddingDtype = useHotVocabularyTables
    ? hotVocabularyRuntime.outputDtype
    : hasSplitEmbeddingTables
    ? getEmbeddingDtype(embedTokensPerLayerSplit[0])
    : getEmbeddingDtype(embedTokensPerLayer);
  const embedSource = hasSplitEmbeddingTables || useHotVocabularyTables
    ? null
    : getEmbeddingSource(embedTokensPerLayer, 'embedTokensPerLayer');
  const totalPerLayerHiddenSize = numLayers * hiddenSizePerLayerInput;
  const hasRangeBackedProjection = isCpuWeightBuffer(perLayerModelProjection)
    && isRangeBackedPleProjectionSource(perLayerModelProjection.data);
  const projectionWeight = hasRangeBackedProjection
    ? null
    : getWeightBuffer(perLayerModelProjection, 'per_layer_model_projection');
  if (isWeightBuffer(perLayerModelProjection) && perLayerModelProjection.layout !== 'row') {
    throw new Error(
      'Gemma 4 per-layer input projection requires a row-major per_layer_model_projection weight. ' +
      `Got layout="${perLayerModelProjection.layout}".`
    );
  }
  const projectionScale = config.hiddenSize ** -0.5;
  const combineScale = 2 ** -0.5;
  const scaledProjectionNormWeight = await ensurePleScaledProjectionNormWeight(context, combineScale);
  const usesCachedScaledProjectionNormWeight = !!scaledProjectionNormWeight;
  const projectionNormWeight = scaledProjectionNormWeight ?? getNormWeightBuffer(
    perLayerProjectionNorm,
    'per_layer_projection_norm',
    weightConfig,
    debugFlags
  );
  const perLayerBuffers = new Array(numLayers).fill(null);
  const pleCache = options.pleCache ?? null;
  const activationBytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', {
    dtype: activationDtype,
  });
  const projectionDevice = getDevice();
  if (projectionDevice == null) {
    throw new Error('Gemma 4 per-layer input projection requires an initialized GPU device.');
  }
  const pleActivationStaticQdq = resolvePleActivationStaticQdqConfig(perLayerInputWeights);
  let projectionInputTensor = inputEmbedsTensor;

  // Decode-path optimizations: coalesced PLE read + fused projection matmul.
  // Gated on numTokens === 1 (decode) and row-major embeddings (non-transpose).
  // For numTokens > 1 (prefill), the fused matmul output is strided per-layer,
  // so we fall back to the per-layer path.
  const embedTranspose = (hasSplitEmbeddingTables || useHotVocabularyTables) ? false : getEmbeddingTranspose(embedTokensPerLayer);
  const canFuseDecodeOps = numTokens === 1 && !embedTranspose && !hasRangeBackedProjection;
  const tokenIdsAreGpuBuffer = isGpuBufferInstance(tokenIds);

  const tokenIdHint = Number.isInteger(options.tokenIdHint) ? Number(options.tokenIdHint) : null;
  const decodeTokenId = canFuseDecodeOps
    ? (tokenIdsAreGpuBuffer
      ? (Number.isInteger(tokenIdHint) ? tokenIdHint : null)
      : Number(tokenIds[0]))
    : null;

  if (decodeTokenId != null) {
    const preparedTokenHit = getPreparedTokenEntry(
      pleCache,
      decodeTokenId,
      context.perLayerInputsSession ?? null,
      activationDtype,
      context.stats ?? null
    );
    if (preparedTokenHit) {
      return preparedTokenHit;
    }
  }

  // Step 5: Use prefetched PLE row if available and token matches.
  // Falls back to inline coalesced read (step 3) otherwise.
  let preloadedCpuRow = null;
  if (canFuseDecodeOps && !perLayerTokenIds) {
    const embedCpuData = !hasSplitEmbeddingTables && isCpuWeightBuffer(embedSource) ? embedSource.data : null;
    if (embedCpuData && isRangeBackedCpuEmbeddingSource(embedCpuData)) {
      if (tokenIdsAreGpuBuffer) {
        throw new Error(
          'Gemma 4 per-layer input decode with range-backed CPU embeddings requires CPU token IDs. ' +
          'Disable batch decode or use GPU-resident per-layer inputs.'
        );
      }
      preloadedCpuRow = await loadRangeBackedPleRow(
        tokenIds[0],
        embedTokensPerLayer,
        totalPerLayerHiddenSize,
        context.perLayerInputsSession ?? null,
        'Coalesced PLE row',
        options.prefetchedRow ?? null
      );
      if (!preloadedCpuRow) {
        throw new Error('Gemma 4 range-backed per-layer input row load returned null unexpectedly.');
      }
    }
  }

  // Step 6: Batched prefill gather. When numTokens > 1 and the PLE source is
  // range-backed + row-major, read all tokens' full PLE rows into a single CPU
  // buffer. This avoids numTokens × numLayers separate loadRange calls during
  // the per-layer embed loop.
  let prefillBatchedRows = null;
  if (!canFuseDecodeOps && numTokens > 1 && !perLayerTokenIds && !getEmbeddingTranspose(embedTokensPerLayer)) {
    const embedCpuData = !hasSplitEmbeddingTables && isCpuWeightBuffer(embedSource) ? embedSource.data : null;
    if (embedCpuData && isRangeBackedCpuEmbeddingSource(embedCpuData)) {
      const tokenIdArray = Array.isArray(tokenIds) ? tokenIds : Array.from(tokenIds);
      prefillBatchedRows = new Float32Array(numTokens * totalPerLayerHiddenSize);
      for (let t = 0; t < numTokens; t++) {
        const row = await loadRangeBackedPleRow(
          tokenIdArray[t],
          embedTokensPerLayer,
          totalPerLayerHiddenSize,
          context.perLayerInputsSession ?? null,
          'Batched PLE prefill row'
        );
        if (!row) {
          throw new Error('Gemma 4 batched per-layer input row load returned null unexpectedly.');
        }
        prefillBatchedRows.set(row, t * totalPerLayerHiddenSize);
      }
    }
  }

  // Fused projection matmul: one dispatch for all layers instead of numLayers dispatches.
  // Produces [numTokens × totalPerLayerHiddenSize], then scales the full output and
  // extracts per-layer slices via GPU buffer copies.
  let fusedProjectionSlices = null;
  const embedDtypeResolved = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: perLayerEmbeddingDtype,
  });

  try {
    if (pleActivationStaticQdq) {
      projectionInputTensor = await applyPleActivationStaticQdq(
        inputEmbedsTensor,
        pleActivationStaticQdq.inputScale,
        recorder,
        decodeBuffers,
        'Gemma 4 per-layer projection input'
      );
    }

    if (canFuseDecodeOps) {
      let fusedProjection = null;
      let fusedProjectionForScale = null;
      let scaledFused = null;
      try {
        // Per-layer-model projection: precision island. Source rows decoded
        // from int4_per_row produce activations whose magnitude before
        // RMSNorm-mediated scale-down can exceed f16 max (~65504). When the
        // matmul kernel does not internally widen its accumulator (RDNA3 is
        // strict, Metal happens to widen), the f16 output saturates to Inf
        // and then NaN. Compute matmul + QDQ + scale in f32 so the dynamic
        // range fits, then narrow to activationDtype after the scale step
        // brings magnitudes back to ~O(1).
        fusedProjection = await doMatmul(
          projectionInputTensor,
          projectionWeight,
          numTokens,
          totalPerLayerHiddenSize,
          config.hiddenSize,
          {
            transposeB: 'auto',
            label: 'per_layer_fused_projection',
            kernelPath: context.kernelPath ?? null,
            role: 'per_layer_model_projection',
            outputDtype: 'f32',
          },
          recorder
        );
        fusedProjectionForScale = pleActivationStaticQdq
          ? await applyPleActivationStaticQdq(
            fusedProjection,
            pleActivationStaticQdq.outputScale,
            recorder,
            decodeBuffers,
            'Gemma 4 per-layer projection output'
          )
          : fusedProjection;
        if (fusedProjectionForScale !== fusedProjection) {
          releaseOrTrack(recorder, fusedProjection.buffer, decodeBuffers);
          fusedProjection = null;
        }

        let scaledFusedF32 = recorder
          ? await recordScale(recorder, fusedProjectionForScale, projectionScale, {
            count: numTokens * totalPerLayerHiddenSize,
          })
          : await runScale(fusedProjectionForScale, projectionScale, {
            count: numTokens * totalPerLayerHiddenSize,
          });
        releaseOrTrack(recorder, fusedProjectionForScale.buffer, decodeBuffers);
        fusedProjectionForScale = null;

        scaledFused = scaledFusedF32.dtype === activationDtype
          ? scaledFusedF32
          : await doCast(scaledFusedF32, activationDtype, recorder);
        if (scaledFused !== scaledFusedF32) {
          releaseOrTrack(recorder, scaledFusedF32.buffer, decodeBuffers);
        }

        // Step 4: Extract per-layer slices via GPU buffer copies (one encoder, one submit).
        // Reuse cached slice buffers when available to avoid per-step pool churn.
        const device = getDevice();
        const sliceBytes = hiddenSizePerLayerInput * activationBytesPerElement;
        const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
        fusedProjectionSlices = new Array(numLayers);
        for (let l = 0; l < numLayers; l++) {
          const sliceBuf = pleCache?.sliceBuffers?.[l] ?? acquireBuffer(sliceBytes, undefined, `L${l}.per_layer_proj_slice`);
          encoder.copyBufferToBuffer(scaledFused.buffer, l * sliceBytes, sliceBuf, 0, sliceBytes);
          fusedProjectionSlices[l] = sliceBuf;
        }
        if (!recorder) {
          device.queue.submit([encoder.finish()]);
        }
        releaseOrTrack(recorder, scaledFused.buffer, decodeBuffers);
        scaledFused = null;
      } catch (error) {
        if (scaledFused) {
          releaseOrTrack(recorder, scaledFused.buffer, decodeBuffers);
        }
        if (fusedProjectionForScale && fusedProjectionForScale !== fusedProjection) {
          releaseOrTrack(recorder, fusedProjectionForScale.buffer, decodeBuffers);
        }
        if (fusedProjection) {
          releaseOrTrack(recorder, fusedProjection.buffer, decodeBuffers);
        }
        throw error;
      }
    }

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const hiddenOffset = layerIdx * hiddenSizePerLayerInput;
      const layerEmbedSource = useHotVocabularyTables
        ? getEmbeddingSource(hotVocabularyRuntime.splitTables[layerIdx], `embedTokensPerLayerHot[L${layerIdx}]`)
        : hasSplitEmbeddingTables
        ? getEmbeddingSource(embedTokensPerLayerSplit[layerIdx], `embedTokensPerLayerSplit[L${layerIdx}]`)
        : embedSource;
      let gatheredTensor = null;
      let scaledProjectionTensor = null;
      let combinedTensor = null;
      try {
        gatheredTensor = await embed(perLayerTokenIds ?? tokenIds, layerEmbedSource, {
          hiddenSize: hiddenSizePerLayerInput,
          vocabSize: useHotVocabularyTables ? (hotVocabularyRuntime.sentinelIndex + 1) : vocabSizePerLayerInput,
          scaleEmbeddings: true,
          probeStage: 'per_layer_embed_out',
          recorder,
          numTokens,
          indexOffset: perLayerTokenIds ? perLayerIndexOffset : indexOffset,
          transpose: embedTranspose,
          debugProbes: context.debugProbes,
          operatorDiagnostics: context.operatorDiagnostics,
          activationDtype,
          embeddingDtype: embedDtypeResolved,
          executionPolicies: context.executionPolicies ?? null,
          inputHiddenSize: (hasSplitEmbeddingTables || useHotVocabularyTables) ? hiddenSizePerLayerInput : totalPerLayerHiddenSize,
          hiddenOffset: (hasSplitEmbeddingTables || useHotVocabularyTables) ? 0 : hiddenOffset,
          preloadedCpuRow,
          preloadedCpuBatchedRows: prefillBatchedRows,
          outputBuffer: canFuseDecodeOps
            ? (pleCache?.gatherSliceBuffers?.[layerIdx] ?? undefined)
            : undefined,
          stats: context.stats ?? null,
        });

        if (fusedProjectionSlices) {
          // Use pre-computed fused projection slice (already scaled).
          scaledProjectionTensor = createTensor(
            fusedProjectionSlices[layerIdx],
            activationDtype,
            [numTokens, hiddenSizePerLayerInput],
            `L${layerIdx}.per_layer_proj_scaled`
          );
          fusedProjectionSlices[layerIdx] = null;
        } else {
          let projectionWeightForLayer = projectionWeight;
          let projectionWeightBufferForLayer = null;
          let projectionWeightOffset = 0;
          let projectedTensor = null;
          let projectionTensorForScale = null;
          try {
            const rangeBackedProjection = await loadRangeBackedPleProjectionSliceBytes(
              perLayerModelProjection,
              layerIdx,
              hiddenSizePerLayerInput,
              config.hiddenSize,
              `L${layerIdx}.per_layer_projection_in`
            );
            if (rangeBackedProjection) {
              projectionWeightBufferForLayer = acquireBuffer(
                rangeBackedProjection.bytes.byteLength,
                undefined,
                `L${layerIdx}.per_layer_projection_in_weight`
              );
              projectionDevice.queue.writeBuffer(
                projectionWeightBufferForLayer,
                0,
                rangeBackedProjection.bytes,
                rangeBackedProjection.bytes.byteOffset,
                rangeBackedProjection.bytes.byteLength
              );
              projectionWeightForLayer = createWeightBuffer(
                projectionWeightBufferForLayer,
                rangeBackedProjection.dtype,
                rangeBackedProjection.layout,
                rangeBackedProjection.shape,
                `L${layerIdx}.per_layer_projection_in_weight`
              );
              projectionWeightOffset = 0;
            } else {
              projectionWeightForLayer = resolveDensePleProjectionWeight(
                projectionWeightForLayer,
                `L${layerIdx}.per_layer_projection_in_weight`
              );
              const projectionWeightDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
                dtype: getPleProjectionWeightDtype(projectionWeightForLayer),
              });
              const projectionWeightBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', {
                dtype: projectionWeightDtype,
              });
              projectionWeightOffset = hiddenOffset * config.hiddenSize * projectionWeightBytes;
            }
            // Per-layer-model projection: precision island (see fused path
            // above for rationale). Force f32 through matmul + QDQ + scale,
            // then narrow to activationDtype after scale brings magnitudes
            // back to ~O(1).
            projectedTensor = await doMatmul(
              projectionInputTensor,
              projectionWeightForLayer,
              numTokens,
              hiddenSizePerLayerInput,
              config.hiddenSize,
              {
                transposeB: 'auto',
                bOffset: projectionWeightOffset,
                label: `L${layerIdx}.per_layer_projection_in`,
                layerIdx,
                kernelPath: context.kernelPath ?? null,
                role: 'per_layer_model_projection',
                outputDtype: 'f32',
              },
              recorder
            );
            await runProbes('per_layer_projection_in', projectedTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize: hiddenSizePerLayerInput,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: 'f32',
            });
            projectionTensorForScale = pleActivationStaticQdq
              ? await applyPleActivationStaticQdq(
                projectedTensor,
                pleActivationStaticQdq.outputScale,
                recorder,
                decodeBuffers,
                `Gemma 4 per-layer projection output L${layerIdx}`
              )
              : projectedTensor;
            if (projectionTensorForScale !== projectedTensor) {
              releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
              projectedTensor = null;
            }
            const scaledF32 = recorder
              ? await recordScale(recorder, projectionTensorForScale, projectionScale, {
                count: numTokens * hiddenSizePerLayerInput,
              })
              : await runScale(projectionTensorForScale, projectionScale, {
                count: numTokens * hiddenSizePerLayerInput,
              });
            if (projectionTensorForScale === projectedTensor) {
              projectedTensor = null;
            }
            releaseOrTrack(recorder, projectionTensorForScale.buffer, decodeBuffers);
            projectionTensorForScale = null;
            scaledProjectionTensor = scaledF32.dtype === activationDtype
              ? scaledF32
              : await doCast(scaledF32, activationDtype, recorder);
            if (scaledProjectionTensor !== scaledF32) {
              releaseOrTrack(recorder, scaledF32.buffer, decodeBuffers);
            }
          } finally {
            if (projectionTensorForScale && projectionTensorForScale !== projectedTensor) {
              releaseOrTrack(recorder, projectionTensorForScale.buffer, decodeBuffers);
            }
            if (projectedTensor) {
              releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
            }
            if (projectionWeightBufferForLayer) {
              releaseOrTrack(recorder, projectionWeightBufferForLayer, decodeBuffers);
            }
          }
          await runProbes('per_layer_projection_scaled', scaledProjectionTensor.buffer, {
            layerIdx,
            numTokens,
            hiddenSize: hiddenSizePerLayerInput,
            probes: context.debugProbes,
            recorder,
            operatorDiagnostics: context.operatorDiagnostics,
            dtype: activationDtype,
          });
        }

        // Fuse the residual add into RMSNorm so Gemma 4 PLE decode avoids an
        // extra dispatch per layer. When the model uses raw RMSNorm weights,
        // cache the fixed combine scale into the norm weights and skip the
        // legacy post-norm scale dispatch entirely.
        combinedTensor = await doRMSNorm(scaledProjectionTensor, projectionNormWeight, config.rmsNormEps, {
          batchSize: numTokens,
          hiddenSize: hiddenSizePerLayerInput,
          residual: gatheredTensor,
          label: `L${layerIdx}.per_layer_input_combine`,
          layerIdx,
          rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
        }, recorder);
        releasePleSliceBuffer(recorder, scaledProjectionTensor.buffer, decodeBuffers, pleCache);
        scaledProjectionTensor = null;
        releasePleSliceBuffer(recorder, gatheredTensor.buffer, decodeBuffers, pleCache);
        gatheredTensor = null;

        if (usesCachedScaledProjectionNormWeight) {
          perLayerBuffers[layerIdx] = combinedTensor.buffer;
          combinedTensor = null;
        } else {
          // Step 8: Inplace scale avoids allocating a separate output buffer.
          // The combined tensor buffer is reused as the final per-layer output.
          const scaledTensor = recorder
            ? await recordScale(recorder, combinedTensor, combineScale, {
              count: numTokens * hiddenSizePerLayerInput,
              inplace: true,
            })
            : await runScale(combinedTensor, combineScale, {
              count: numTokens * hiddenSizePerLayerInput,
              inplace: true,
            });
          perLayerBuffers[layerIdx] = scaledTensor.buffer;
          combinedTensor = null;
        }
        await runProbes('per_layer_input', perLayerBuffers[layerIdx], {
          layerIdx,
          numTokens,
          hiddenSize: hiddenSizePerLayerInput,
          probes: context.debugProbes,
          recorder,
          operatorDiagnostics: context.operatorDiagnostics,
          dtype: activationDtype,
        });
      } catch (error) {
        if (combinedTensor) {
          releaseOrTrack(recorder, combinedTensor.buffer, decodeBuffers);
        }
        if (gatheredTensor) {
          releasePleSliceBuffer(recorder, gatheredTensor.buffer, decodeBuffers, pleCache);
        }
        if (scaledProjectionTensor) {
          releasePleSliceBuffer(recorder, scaledProjectionTensor.buffer, decodeBuffers, pleCache);
        }
        throw error;
      }
    }
  } catch (error) {
    if (fusedProjectionSlices) {
      for (let i = 0; i < fusedProjectionSlices.length; i++) {
        const buf = fusedProjectionSlices[i];
        releasePleSliceBuffer(recorder, buf, decodeBuffers, pleCache);
      }
    }
    for (const buffer of perLayerBuffers) {
      if (buffer) {
        releaseOrTrack(recorder, buffer, decodeBuffers);
      }
    }
    throw error;
  } finally {
    if (projectionInputTensor !== inputEmbedsTensor) {
      releaseOrTrack(recorder, projectionInputTensor.buffer, decodeBuffers);
    }
    if (!usesCachedScaledProjectionNormWeight && !isGpuBufferInstance(perLayerProjectionNorm)) {
      releaseOrTrack(recorder, projectionNormWeight, decodeBuffers);
    }
  }

  if (decodeTokenId != null) {
    return storePreparedTokenEntry(
      pleCache,
      decodeTokenId,
      perLayerBuffers,
      context.perLayerInputsSession ?? null,
      activationDtype,
      context.stats ?? null
    );
  }

  // Prefill hot-cache seeding: when we computed a multi-token batch and pleCache is
  // provided, extract per-token row slices and store unique token IDs in the cache.
  // Copies are recorded via the active recorder (if any) so they execute after the
  // batch computation and before the first decode step reads from the cache.
  // Without a recorder, the batch buffers are already populated so copies are safe
  // to submit immediately.
  if (pleCache) {
    const plePolicy = getPleHotCachePolicy(context.perLayerInputsSession ?? null);
    if (plePolicy && plePolicy.mode === 'prepared_tokens'
        && pleCache.preparedTokenEntries instanceof Map
        && !tokenIdsAreGpuBuffer
        && numTokens > 0
    ) {
      const tokenIdArray = Array.isArray(tokenIds) ? tokenIds : Array.from(tokenIds);
      const sliceBytes = hiddenSizePerLayerInput * activationBytesPerElement;
      const seen = new Set();
      const device = getDevice();
      let pendingEncoder = null;
      for (let tokenPos = 0; tokenPos < tokenIdArray.length; tokenPos++) {
        const tid = tokenIdArray[tokenPos];
        if (seen.has(tid) || pleCache.preparedTokenEntries.has(tid)) continue;
        if (pleCache.preparedTokenEntries.size >= plePolicy.maxTokens) break;
        seen.add(tid);
        const srcOffset = tokenPos * sliceBytes;
        const sliceBuffers = new Array(numLayers).fill(null);
        let allValid = true;
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
          const srcBuffer = perLayerBuffers[layerIdx];
          if (!srcBuffer || srcBuffer.size < srcOffset + sliceBytes) {
            allValid = false;
            break;
          }
          const dstBuffer = acquireBuffer(sliceBytes, undefined, `ple_seed_L${layerIdx}`);
          if (recorder) {
            recorder.getEncoder().copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, 0, sliceBytes);
          } else {
            if (!pendingEncoder) pendingEncoder = device.createCommandEncoder();
            pendingEncoder.copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, 0, sliceBytes);
          }
          sliceBuffers[layerIdx] = dstBuffer;
        }
        if (!allValid) {
          for (const buf of sliceBuffers) {
            if (buf) releaseBuffer(buf);
          }
          continue;
        }
        storePreparedTokenEntry(
          pleCache, tid, sliceBuffers,
          context.perLayerInputsSession ?? null,
          activationDtype,
          context.stats ?? null
        );
      }
      if (pendingEncoder && device) {
        device.queue.submit([pendingEncoder.finish()]);
      }
    }
  }

  return perLayerBuffers;
}

export function createPerLayerInputTensor(buffer, numTokens, hiddenSizePerLayerInput, activationDtype) {
  return createTensor(
    buffer,
    activationDtype,
    [numTokens, hiddenSizePerLayerInput],
    'per_layer_input'
  );
}
