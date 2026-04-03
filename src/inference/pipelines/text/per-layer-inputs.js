import { getWeightDtype, isCpuWeightBuffer, isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { createTensor } from '../../../gpu/tensor.js';
import { recordScale, runScale } from '../../../gpu/kernel-selector.js';
import { getNormWeightBuffer, getWeightBuffer } from './weights.js';
import { doMatmul, doResidualAdd, doRMSNorm, releaseOrTrack } from './ops.js';
import { embed } from './embed.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

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
  const projectionNormWeight = getNormWeightBuffer(
    perLayerProjectionNorm,
    'per_layer_projection_norm',
    weightConfig,
    debugFlags
  );
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
  const perLayerBuffers = new Array(numLayers).fill(null);

  try {
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const hiddenOffset = layerIdx * hiddenSizePerLayerInput;
      const gatheredTensor = await embed(tokenIds, embedSource, {
        hiddenSize: hiddenSizePerLayerInput,
        vocabSize: vocabSizePerLayerInput,
        scaleEmbeddings: true,
        recorder,
        numTokens,
        indexOffset,
        transpose: getEmbeddingTranspose(embedTokensPerLayer),
        debugProbes: context.debugProbes,
        operatorDiagnostics: context.operatorDiagnostics,
        activationDtype,
        embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
          dtype: perLayerEmbeddingDtype,
        }),
        inputHiddenSize: totalPerLayerHiddenSize,
        hiddenOffset,
      });

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
      projectedTensor = recorder
        ? await recordScale(recorder, projectedTensor, projectionScale, {
          count: numTokens * hiddenSizePerLayerInput,
          inplace: true,
        })
        : await runScale(projectedTensor, projectionScale, {
          count: numTokens * hiddenSizePerLayerInput,
          inplace: true,
        });

      const normalizedTensor = await doRMSNorm(projectedTensor, projectionNormWeight, config.rmsNormEps, {
        batchSize: numTokens,
        hiddenSize: hiddenSizePerLayerInput,
        label: `L${layerIdx}.per_layer_projection_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
      releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);

      const combinedTensor = await doResidualAdd(
        normalizedTensor,
        gatheredTensor,
        numTokens * hiddenSizePerLayerInput,
        recorder,
        {
          label: `L${layerIdx}.per_layer_input_combine`,
          layerIdx,
        }
      );
      releaseOrTrack(recorder, normalizedTensor.buffer, decodeBuffers);
      releaseOrTrack(recorder, gatheredTensor.buffer, decodeBuffers);

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
    }
  } catch (error) {
    for (const buffer of perLayerBuffers) {
      if (buffer) {
        releaseOrTrack(recorder, buffer, decodeBuffers);
      }
    }
    throw error;
  } finally {
    if (!isGpuBufferInstance(perLayerProjectionNorm)) {
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
