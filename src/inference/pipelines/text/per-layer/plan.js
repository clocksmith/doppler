import {
  createWeightBuffer,
  getBufferDtype,
  getLayout,
  getWeightDtype,
  isCpuWeightBuffer,
  isGpuBufferInstance,
  isWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  castF16ToF32,
  castF32ToF16,
  recordActivationStaticQdq,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordScale,
  runActivationStaticQdq,
  runScale,
} from '../../../../gpu/kernel-selector.js';
import { getNormWeightBuffer, getWeightBuffer } from '../weights.js';
import { doCast, doMatmul, doRMSNorm, releaseOrTrack } from '../ops.js';
import { runProbes } from '../probes.js';
import { embed, isRangeBackedCpuEmbeddingSource, normalizeRangeBytes, decodeRangeChunkIntoOutput } from '../embed.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { f16ToF32 } from '../../../../loader/dtype-utils.js';
import { destroyPleHotVocabularyRuntime, destroyPleRuntimeCacheEntry, ensurePleScaledProjectionNormWeight, getPerLayerInputWeights, getPleHotCachePolicy, getPleRangeRowLoadConfig, getPleSplitTablePolicy, pleRuntimeCache } from './materialize.js';

export const pleRangeRowCache = new WeakMap();

export const PLE_ACTIVATION_STATIC_QMIN = -128;

export const PLE_ACTIVATION_STATIC_QMAX = 127;

export function isRangeBackedPleProjectionSource(value) {
  return isRangeBackedCpuEmbeddingSource(value);
}

export function getPleProjectionWeightDtype(weight) {
  if (isCpuWeightBuffer(weight)) {
    return weight.dtype ?? null;
  }
  return getWeightDtype(weight);
}

export function getTensorElementCount(tensor, label) {
  if (!Array.isArray(tensor?.shape) || tensor.shape.length === 0) {
    throw new Error(`${label} requires a tensor with an explicit shape.`);
  }
  const count = tensor.shape.reduce((total, dimension) => total * Number(dimension), 1);
  if (!Number.isFinite(count) || count <= 0) {
    throw new Error(`${label} resolved an invalid tensor element count (${String(count)}).`);
  }
  return count;
}

export async function applyPleActivationStaticQdq(tensor, scale, recorder, decodeBuffers, label) {
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

export function getPleRowCachePolicy(sessionConfig) {
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

export function releasePreparedTokenEntry(cache, tokenId, entry) {
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

export function prunePreparedTokenCache(cache, policy) {
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

export function getPreparedTokenEntry(cache, tokenId, sessionConfig, activationDtype, stats = null) {
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

export function storePreparedTokenEntry(cache, tokenId, buffers, sessionConfig, activationDtype, stats = null) {
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

export function getPleRangeRowCache(embedTokensPerLayer, sessionConfig) {
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

export function touchPleCachedRow(cache, tokenId) {
  const hit = cache?.rows?.get(tokenId) ?? null;
  if (!hit) {
    return null;
  }
  cache.rows.delete(tokenId);
  cache.rows.set(tokenId, hit);
  return hit.row;
}

export function prunePleRangeRowCache(cache) {
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

export function cachePleRangeRow(cache, tokenId, row) {
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

export async function loadRangeBackedPleRow(
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
