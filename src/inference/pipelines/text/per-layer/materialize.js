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

export const pleRuntimeCache = new WeakMap();

export let f16ToF32Lookup = null;

export function getF16ToF32Lookup() {
  if (f16ToF32Lookup) {
    return f16ToF32Lookup;
  }
  const lookup = new Float32Array(1 << 16);
  for (let value = 0; value < lookup.length; value += 1) {
    lookup[value] = f16ToF32(value);
  }
  f16ToF32Lookup = lookup;
  return lookup;
}

export function getPerLayerInputWeights(context) {
  const weights = context.weights.get('per_layer_inputs');
  if (!weights || typeof weights !== 'object') {
    throw new Error(
      'Gemma 4 per-layer inputs require global per-layer input weights, ' +
      'but state.weights.get("per_layer_inputs") was missing.'
    );
  }
  return weights;
}

export function normalizePleProjectionNormDtype(dtype) {
  if (typeof dtype !== 'string') {
    return null;
  }
  const value = dtype.toLowerCase();
  if (value === 'f16' || value === 'f32') {
    return value;
  }
  return null;
}

export function getPleProjectionNormDtype(weight) {
  return normalizePleProjectionNormDtype(getWeightDtype(weight))
    ?? normalizePleProjectionNormDtype(getBufferDtype(weight))
    ?? null;
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

export function createPleProjectionNormTensor(buffer, dtype, hiddenSizePerLayerInput, label) {
  return createTensor(buffer, dtype, [hiddenSizePerLayerInput], label);
}

export function destroyPleRuntimeCacheEntry(entry) {
  const cachedBuffer = entry?.scaledProjectionNormWeight?.buffer ?? null;
  if (cachedBuffer) {
    releaseBuffer(cachedBuffer);
  }
}

export function destroyPleHotVocabularyRuntime(perLayerInputWeights) {
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
      inferPleProjectionNormDtype(projectionNormWeight, hiddenSizePerLayerInput),
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

export function requirePleSourceDtype(rawDtype, label, allowed = ['f16', 'f32']) {
  if (rawDtype == null) {
    throw new Error(`${label} requires source dtype metadata.`);
  }
  const sourceDtype = String(rawDtype).toLowerCase();
  if (!allowed.includes(sourceDtype)) {
    throw new Error(`${label} requires ${allowed.join('/')} source rows; got "${sourceDtype}".`);
  }
  return sourceDtype;
}

export function getPleHotCachePolicy(sessionConfig) {
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

export function resolvePleHotVocabularyCapacity({
  maxTokens,
  maxBytes,
  numLayers,
  hiddenSize,
  bytesPerElement,
  vocabSize,
}) {
  const bytesPerHotRow = numLayers * hiddenSize * bytesPerElement;
  const tokenIndexMapBytes = vocabSize * Uint32Array.BYTES_PER_ELEMENT;
  const tableBudgetBytes = Math.max(0, maxBytes - tokenIndexMapBytes);
  const maxTableRows = bytesPerHotRow > 0
    ? Math.floor(tableBudgetBytes / bytesPerHotRow)
    : 0;
  const maxHotTokens = Math.max(0, Math.min(maxTokens, maxTableRows - 1));
  return {
    maxHotTokens,
    bytesPerHotRow,
    tokenIndexMapBytes,
  };
}

export function getPleSplitTablePolicy(sessionConfig) {
  if (sessionConfig?.materialization !== 'gpu_split_tables') {
    return null;
  }
  return { mode: 'gpu_split_tables' };
}

export function getPleRangeRowLoadConfig(embedTokensPerLayer, totalPerLayerHiddenSize) {
  const sourceDtype = requirePleSourceDtype(
    (isCpuWeightBuffer(embedTokensPerLayer) ? embedTokensPerLayer.data?.sourceDtype : null)
      ?? embedTokensPerLayer?.dtype,
    'Gemma 4 range-backed per-layer input rows',
    ['f16', 'bf16', 'f32']
  );
  const bytesPerElement = (sourceDtype === 'f16' || sourceDtype === 'bf16') ? 2 : 4;
  return {
    sourceDtype,
    sourceRowBytes: totalPerLayerHiddenSize * bytesPerElement,
  };
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

  const sourceDtype = requirePleSourceDtype(
    embedTokensPerLayer.data?.sourceDtype ?? embedTokensPerLayer.dtype,
    'Gemma 4 gpu_split_tables materialization'
  );

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

export function resolvePleHotVocabularySeedTokenIds(context, maxTokens, vocabSizePerLayerInput) {
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

export function mergePleHotVocabularyTokenIds(seedTokenIds, tokenizerHotTokenIds, maxTokens, vocabSizePerLayerInput) {
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

  const bytesPerElement = policy.outputDtype === 'f16' ? 2 : 4;
  const capacity = resolvePleHotVocabularyCapacity({
    maxTokens: policy.maxTokens,
    maxBytes: policy.maxBytes,
    numLayers,
    hiddenSize: hiddenSizePerLayerInput,
    bytesPerElement,
    vocabSize: vocabSizePerLayerInput,
  });
  if (capacity.maxHotTokens <= 0) {
    return null;
  }

  const tokenizer = context?.tokenizer ?? null;
  const tokenizerHotTokenIds = typeof tokenizer?.getHotTokenIds === 'function'
    ? tokenizer.getHotTokenIds(capacity.maxHotTokens)
    : null;
  const seedTokenIds = resolvePleHotVocabularySeedTokenIds(
    context,
    capacity.maxHotTokens,
    vocabSizePerLayerInput
  );
  const hotTokenIds = mergePleHotVocabularyTokenIds(
    seedTokenIds,
    tokenizerHotTokenIds,
    capacity.maxHotTokens,
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
    && cached.maxTokens === capacity.maxHotTokens
    && cached.maxBytes === policy.maxBytes
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

  const sourceDtype = requirePleSourceDtype(
    embedTokensPerLayer.data?.sourceDtype ?? embedTokensPerLayer.dtype,
    'Gemma 4 hot vocabulary cache'
  );
  const expandsF16ToF32 = sourceDtype === 'f16' && policy.outputDtype === 'f32';
  if (sourceDtype !== policy.outputDtype && !expandsF16ToF32) {
    throw new Error(
      `Gemma 4 hot vocabulary cache cannot convert source dtype "${sourceDtype}" ` +
      `to output dtype "${policy.outputDtype}".`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Gemma 4 hot vocabulary cache.');
  }

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
    const expandedF32Row = expandsF16ToF32
      ? new Float32Array(totalPerLayerHiddenSize)
      : null;
    const f16Lookup = expandsF16ToF32 ? getF16ToF32Lookup() : null;
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
      } else if (sourceDtype === 'f16') {
        const sourceWords = new Uint16Array(chunk.buffer, chunk.byteOffset, totalPerLayerHiddenSize);
        for (let valueIdx = 0; valueIdx < totalPerLayerHiddenSize; valueIdx += 1) {
          expandedF32Row[valueIdx] = f16Lookup[sourceWords[valueIdx]];
        }
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
          const sourceStart = layerIdx * hiddenSizePerLayerInput;
          device.queue.writeBuffer(
            splitTables[layerIdx].buffer,
            hotIndex * hiddenSizePerLayerInput * bytesPerElement,
            expandedF32Row.buffer,
            sourceStart * bytesPerElement,
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
    maxTokens: capacity.maxHotTokens,
    maxBytes: policy.maxBytes,
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
