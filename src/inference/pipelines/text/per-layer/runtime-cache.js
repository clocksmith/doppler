import { getWeightDtype, isCpuWeightBuffer, isGpuBufferInstance, isWeightBuffer } from "../../../../gpu/weight-buffer.js";
import { releaseOrTrack } from "../execution/ops.js";
import { isRangeBackedCpuEmbeddingSource } from "../embedding/embed.js";
import { acquireBuffer, releaseBuffer } from "../../../../memory/buffer-pool.js";
import { destroyPleHotVocabularyRuntime, destroyPleRuntimeCacheEntry, getPleSplitTablePolicy, pleRuntimeCache } from "./materialize.js";
import { loadRangeBackedPleRow, pleRangeRowCache } from "./plan.js";

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

export function getEmbeddingSource(weight, label) {
  if (isWeightBuffer(weight)) {
    return weight.buffer;
  }
  if (isCpuWeightBuffer(weight) || isGpuBufferInstance(weight) || weight instanceof Float32Array) {
    return weight;
  }
  throw new Error(`Gemma 4 per-layer input ${label} has unsupported type "${weight?.constructor?.name ?? typeof weight}".`);
}

export function getEmbeddingDtype(weight) {
  if (isCpuWeightBuffer(weight)) {
    return weight.dtype;
  }
  return getWeightDtype(weight);
}

export function getEmbeddingTranspose(weight) {
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

export function releasePleSliceBuffer(recorder, buffer, decodeBuffers, cache) {
  if (!buffer || isCachedPleSliceBuffer(cache, buffer)) {
    return;
  }
  releaseOrTrack(recorder, buffer, decodeBuffers);
}

export function getPleHotVocabularyRuntime(context) {
  const perLayerInputWeights = context?.weights?.get?.('per_layer_inputs');
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return null;
  }
  const runtime = perLayerInputWeights.embedTokensPerLayerHotRuntime ?? null;
  return runtime && typeof runtime === 'object' ? runtime : null;
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

  const sessionConfig = context?.perLayerInputsSession ?? context?.config?.perLayerInputsSession ?? null;
  if (getPleSplitTablePolicy(sessionConfig)) {
    return true;
  }

  const perLayerInputWeights = context?.weights?.get?.('per_layer_inputs');
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return false;
  }

  return Array.isArray(perLayerInputWeights.embedTokensPerLayerSplit)
    && perLayerInputWeights.embedTokensPerLayerSplit.length > 0;
}
