import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { createTensor } from '../../../gpu/tensor.js';
import { recordScale, runScale } from '../../../gpu/kernel-selector.js';
import { getNormWeightBuffer, getWeightBuffer } from './weights.js';
import { doMatmul, doRMSNorm, releaseOrTrack } from './ops.js';
import { embed, isRangeBackedCpuEmbeddingSource, normalizeRangeBytes, decodeRangeChunkIntoOutput } from './embed.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';

const pleRuntimeCache = new WeakMap();

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

export function destroyPleRuntimeCache(perLayerInputWeights) {
  if (!perLayerInputWeights || typeof perLayerInputWeights !== 'object') {
    return;
  }
  const entry = pleRuntimeCache.get(perLayerInputWeights);
  if (!entry) {
    return;
  }
  destroyPleRuntimeCacheEntry(entry);
  pleRuntimeCache.delete(perLayerInputWeights);
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
  return {
    sliceBuffers: Array.from({ length: numLayers }, (_, l) =>
      acquireBuffer(sliceBytes, undefined, `L${l}.ple_slice_cached`)),
  };
}

export function destroyPleBufferCache(cache) {
  if (!cache?.sliceBuffers) return;
  for (const buf of cache.sliceBuffers) {
    if (buf) releaseBuffer(buf);
  }
  cache.sliceBuffers = null;
}

function isCachedPleSliceBuffer(cache, buffer) {
  return Array.isArray(cache?.sliceBuffers) && cache.sliceBuffers.includes(buffer);
}

function releasePleSliceBuffer(recorder, buffer, decodeBuffers, cache) {
  if (!buffer || isCachedPleSliceBuffer(cache, buffer)) {
    return;
  }
  releaseOrTrack(recorder, buffer, decodeBuffers);
}

// Step 5: Prefetch next token's PLE row during current decode step.
// Returns a promise resolving to { tokenId, row: Float32Array } or null.
// Call after sampling produces the next token; pass result as options.prefetchedRow
// to the next preparePerLayerInputs call.
export function prefetchPerLayerRow(tokenId, embedTokensPerLayer, totalPerLayerHiddenSize) {
  if (!isCpuWeightBuffer(embedTokensPerLayer)) return null;
  const cpuData = embedTokensPerLayer.data;
  if (!isRangeBackedCpuEmbeddingSource(cpuData)) return null;
  const sourceDtype = String(cpuData.sourceDtype ?? embedTokensPerLayer.dtype ?? 'f32').toLowerCase();
  const bpe = (sourceDtype === 'f16' || sourceDtype === 'bf16') ? 2 : 4;
  const rowBytes = totalPerLayerHiddenSize * bpe;
  return cpuData.loadRange(tokenId * totalPerLayerHiddenSize * bpe, rowBytes)
    .then(raw => {
      const chunk = normalizeRangeBytes(raw, 'Prefetched PLE row');
      const row = new Float32Array(totalPerLayerHiddenSize);
      decodeRangeChunkIntoOutput(chunk, sourceDtype, row, 0, totalPerLayerHiddenSize);
      return { tokenId, row };
    })
    .catch(() => null);
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
  const perLayerModelProjection = perLayerInputWeights.perLayerModelProjection;
  const perLayerProjectionNorm = perLayerInputWeights.perLayerProjectionNorm;
  if (!embedTokensPerLayer || !perLayerModelProjection || !perLayerProjectionNorm) {
    throw new Error(
      'Gemma 4 per-layer inputs require embedTokensPerLayer, perLayerModelProjection, ' +
      'and perLayerProjectionNorm weights.'
    );
  }

  const numLayers = config.numLayers;
  const numTokens = Number.isFinite(options.numTokens) ? options.numTokens : inputEmbedsTensor.shape?.[0];
  const indexOffset = Number.isFinite(options.indexOffset) ? options.indexOffset : 0;
  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('Gemma 4 per-layer inputs require a positive numTokens value.');
  }

  const activationDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: inputEmbedsTensor.dtype,
  });
  const perLayerEmbeddingDtype = getEmbeddingDtype(embedTokensPerLayer);
  const embedSource = getEmbeddingSource(embedTokensPerLayer, 'embedTokensPerLayer');
  const totalPerLayerHiddenSize = numLayers * hiddenSizePerLayerInput;
  const projectionWeight = getWeightBuffer(perLayerModelProjection, 'per_layer_model_projection');
  if (isWeightBuffer(perLayerModelProjection) && perLayerModelProjection.layout !== 'row') {
    throw new Error(
      'Gemma 4 per-layer input projection requires a row-major per_layer_model_projection weight. ' +
      `Got layout="${perLayerModelProjection.layout}".`
    );
  }
  const projectionWeightDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: getWeightDtype(perLayerModelProjection),
  });
  const projectionWeightBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', {
    dtype: projectionWeightDtype,
  });
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

  // Decode-path optimizations: coalesced PLE read + fused projection matmul.
  // Gated on numTokens === 1 (decode) and row-major embeddings (non-transpose).
  // For numTokens > 1 (prefill), the fused matmul output is strided per-layer,
  // so we fall back to the per-layer path.
  const canFuseDecodeOps = numTokens === 1 && !getEmbeddingTranspose(embedTokensPerLayer);

  // Step 5: Use prefetched PLE row if available and token matches.
  // Falls back to inline coalesced read (step 3) otherwise.
  let preloadedCpuRow = null;
  if (canFuseDecodeOps) {
    const prefetched = options.prefetchedRow;
    if (prefetched && prefetched.tokenId === tokenIds[0]) {
      preloadedCpuRow = prefetched.row;
    } else {
      const embedCpuData = isCpuWeightBuffer(embedSource) ? embedSource.data : null;
      if (embedCpuData && isRangeBackedCpuEmbeddingSource(embedCpuData)) {
        const tokenId = tokenIds[0];
        const sourceDtype = String(embedCpuData.sourceDtype ?? embedSource.dtype ?? 'f32').toLowerCase();
        const bpe = (sourceDtype === 'f16' || sourceDtype === 'bf16') ? 2 : 4;
        const rowBytes = totalPerLayerHiddenSize * bpe;
        const chunk = normalizeRangeBytes(
          await embedCpuData.loadRange(tokenId * totalPerLayerHiddenSize * bpe, rowBytes),
          'Coalesced PLE row'
        );
        preloadedCpuRow = new Float32Array(totalPerLayerHiddenSize);
        decodeRangeChunkIntoOutput(chunk, sourceDtype, preloadedCpuRow, 0, totalPerLayerHiddenSize);
      }
    }
  }

  // Step 6: Batched prefill gather. When numTokens > 1 and the PLE source is
  // range-backed + row-major, read all tokens' full PLE rows into a single CPU
  // buffer. This avoids numTokens × numLayers separate loadRange calls during
  // the per-layer embed loop.
  let prefillBatchedRows = null;
  if (!canFuseDecodeOps && numTokens > 1 && !getEmbeddingTranspose(embedTokensPerLayer)) {
    const embedCpuData = isCpuWeightBuffer(embedSource) ? embedSource.data : null;
    if (embedCpuData && isRangeBackedCpuEmbeddingSource(embedCpuData)) {
      const sourceDtype = String(embedCpuData.sourceDtype ?? embedSource.dtype ?? 'f32').toLowerCase();
      const bpe = (sourceDtype === 'f16' || sourceDtype === 'bf16') ? 2 : 4;
      const rowBytes = totalPerLayerHiddenSize * bpe;
      const tokenIdArray = Array.isArray(tokenIds) ? tokenIds : Array.from(tokenIds);
      prefillBatchedRows = new Float32Array(numTokens * totalPerLayerHiddenSize);
      for (let t = 0; t < numTokens; t++) {
        const chunk = normalizeRangeBytes(
          await embedCpuData.loadRange(tokenIdArray[t] * totalPerLayerHiddenSize * bpe, rowBytes),
          'Batched PLE prefill row'
        );
        decodeRangeChunkIntoOutput(chunk, sourceDtype, prefillBatchedRows, t * totalPerLayerHiddenSize, totalPerLayerHiddenSize);
      }
    }
  }

  // Fused projection matmul: one dispatch for all layers instead of numLayers dispatches.
  // Produces [numTokens × totalPerLayerHiddenSize], then scales the full output and
  // extracts per-layer slices via GPU buffer copies.
  let fusedProjectionSlices = null;
  if (canFuseDecodeOps) {
    const fusedProjection = await doMatmul(
      inputEmbedsTensor,
      projectionWeight,
      numTokens,
      totalPerLayerHiddenSize,
      config.hiddenSize,
      {
        transposeB: 'auto',
        label: 'per_layer_fused_projection',
        kernelPath: context.kernelPath ?? null,
        role: 'per_layer_model_projection',
        outputDtype: activationDtype,
      },
      recorder
    );

    // One scale dispatch instead of numLayers
    const scaledFused = recorder
      ? await recordScale(recorder, fusedProjection, projectionScale, {
        count: numTokens * totalPerLayerHiddenSize,
      })
      : await runScale(fusedProjection, projectionScale, {
        count: numTokens * totalPerLayerHiddenSize,
      });
    releaseOrTrack(recorder, fusedProjection.buffer, decodeBuffers);

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
  }

  const embedDtypeResolved = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: perLayerEmbeddingDtype,
  });
  const embedTranspose = getEmbeddingTranspose(embedTokensPerLayer);

  try {
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const hiddenOffset = layerIdx * hiddenSizePerLayerInput;
      let gatheredTensor = null;
      let scaledProjectionTensor = null;
      let combinedTensor = null;
      try {
        gatheredTensor = await embed(tokenIds, embedSource, {
          hiddenSize: hiddenSizePerLayerInput,
          vocabSize: vocabSizePerLayerInput,
          scaleEmbeddings: true,
          recorder,
          numTokens,
          indexOffset,
          transpose: embedTranspose,
          debugProbes: context.debugProbes,
          operatorDiagnostics: context.operatorDiagnostics,
          activationDtype,
          embeddingDtype: embedDtypeResolved,
          inputHiddenSize: totalPerLayerHiddenSize,
          hiddenOffset,
          preloadedCpuRow,
          preloadedCpuBatchedRows: prefillBatchedRows,
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
          let projectedTensor = await doMatmul(
            inputEmbedsTensor,
            projectionWeight,
            numTokens,
            hiddenSizePerLayerInput,
            config.hiddenSize,
            {
              transposeB: 'auto',
              bOffset: hiddenOffset * config.hiddenSize * projectionWeightBytes,
              label: `L${layerIdx}.per_layer_projection_in`,
              layerIdx,
              kernelPath: context.kernelPath ?? null,
              role: 'per_layer_model_projection',
              outputDtype: activationDtype,
            },
            recorder
          );
          scaledProjectionTensor = recorder
            ? await recordScale(recorder, projectedTensor, projectionScale, {
              count: numTokens * hiddenSizePerLayerInput,
            })
            : await runScale(projectedTensor, projectionScale, {
              count: numTokens * hiddenSizePerLayerInput,
            });
          releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
          projectedTensor = null;
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
        releaseOrTrack(recorder, gatheredTensor.buffer, decodeBuffers);
        gatheredTensor = null;

        if (usesCachedScaledProjectionNormWeight) {
          perLayerBuffers[layerIdx] = combinedTensor.buffer;
          combinedTensor = null;
          continue;
        }

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
      } catch (error) {
        if (combinedTensor) {
          releaseOrTrack(recorder, combinedTensor.buffer, decodeBuffers);
        }
        if (gatheredTensor) {
          releaseOrTrack(recorder, gatheredTensor.buffer, decodeBuffers);
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
    if (!usesCachedScaledProjectionNormWeight && !isGpuBufferInstance(perLayerProjectionNorm)) {
      releaseOrTrack(recorder, projectionNormWeight, decodeBuffers);
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
