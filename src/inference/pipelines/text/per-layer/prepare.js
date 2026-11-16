import { createWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isWeightBuffer } from "../../../../gpu/weight-buffer.js";
import { createTensor } from "../../../../gpu/tensor.js";
import { recordScale, runScale } from "../../../../gpu/kernels/scale.js";
import { getNormWeightBuffer, getWeightBuffer } from "../loading/weights.js";
import { doCast, doMatmul, doRMSNorm, releaseOrTrack } from "../execution/ops.js";
import { runProbes } from "../observation/probes.js";
import { embed, isRangeBackedCpuEmbeddingSource } from "../embedding/embed.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";
import { getDevice } from "../../../../gpu/device.js";
import { acquireBuffer, releaseBuffer } from "../../../../memory/buffer-pool.js";
import { ensurePleScaledProjectionNormWeight, getPerLayerInputWeights, getPleHotCachePolicy } from "./materialize.js";
import { applyPleActivationStaticQdq, getPleProjectionWeightDtype, getPreparedTokenEntry, isRangeBackedPleProjectionSource, loadRangeBackedPleProjectionSliceBytes, loadRangeBackedPleRow, storePreparedTokenEntry } from "./plan.js";
import { getPleHotVocabularyRuntime, getEmbeddingDtype, getEmbeddingSource, getEmbeddingTranspose, releasePleSliceBuffer } from "./runtime-cache.js";
import { resolvePleActivationStaticQdqConfig, resolveDensePleProjectionWeight } from "./projection-contract.js";

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
          embeddingScale: null,
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
