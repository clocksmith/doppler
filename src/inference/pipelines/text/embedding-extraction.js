import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import { createWeightBuffer, requireWeightDtype } from '../../../gpu/weight-buffer.js';
import { acquireBuffer, readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { runLayerNorm, runMatmul, runRMSNorm } from '../../../gpu/kernel-selector.js';
import { runBiasAdd } from '../../../gpu/kernels/residual.js';
import { runEmbeddingPool } from '../../../gpu/kernels/embedding-pool.js';
import { runL2Normalize } from '../../../gpu/kernels/l2-normalize.js';
import { decodeReadback } from './debug-utils/index.js';
import { resolveFinalNormGpuBuffer } from './logits/gpu-executor.js';

function buildSequencePoolControl(tokenIds, pooling) {
  const excluded = new Set(pooling.excludeTokenIds);
  const mask = new Uint32Array(tokenIds.length);
  const tokenMask = new Uint8Array(tokenIds.length);
  let includedCount = 0;
  let lastIndex = -1;
  for (let index = 0; index < tokenIds.length; index += 1) {
    const included = !excluded.has(tokenIds[index]);
    mask[index] = included ? 1 : 0;
    tokenMask[index] = included ? 1 : 0;
    if (included) {
      includedCount += 1;
      lastIndex = index;
    }
  }
  if (includedCount === 0) {
    throw new Error('Sequence pooling excluded every input token.');
  }
  return { mask, tokenMask, includedCount, lastIndex };
}

async function readTensorOutput(tensor) {
  const elementCount = tensor.shape.reduce((product, value) => product * value, 1);
  const bytesPerElement = tensor.dtype === 'f16' ? 2 : 4;
  const data = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
  return decodeReadback(data, tensor.dtype);
}

function uploadFinalNormBias(device, bias, hiddenSize, config) {
  if (config.finalNormBiasTensor !== null && !(bias instanceof Float32Array)) {
    throw new Error(
      `[Pipeline] LayerNorm declares bias tensor "${config.finalNormBiasTensor}" but it was not loaded.`
    );
  }
  const values = bias instanceof Float32Array ? bias : new Float32Array(hiddenSize);
  if (values.length !== hiddenSize) {
    throw new Error(
      `[Pipeline] LayerNorm final bias length must be ${hiddenSize}; got ${values.length}.`
    );
  }
  const buffer = acquireBuffer(values.byteLength, undefined, 'embedding_final_norm_bias');
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

async function applyEmbeddingPostprocessor(input, postprocessor, config) {
  if (!postprocessor) return input;
  const device = getDevice();
  let current = input;
  try {
    for (let index = 0; index < postprocessor.projections.length; index += 1) {
      const projection = postprocessor.projections[index];
      if (current.shape[1] !== projection.inputSize) {
        throw new Error(
          `[Pipeline] Embedding postprocessor projection ${index} expected inputSize=${projection.inputSize}, got ${current.shape[1]}.`
        );
      }
      if (!(projection.weight instanceof Float32Array)
        || projection.weight.length !== projection.outputSize * projection.inputSize) {
        throw new Error(`[Pipeline] Embedding postprocessor projection ${index} has invalid weight shape.`);
      }
      if (projection.activation !== 'identity') {
        throw new Error(
          `[Pipeline] Unsupported embedding postprocessor activation "${projection.activation}" at projection ${index}.`
        );
      }
      const weightBuffer = acquireBuffer(
        projection.weight.byteLength,
        undefined,
        `embedding_projection_${index}_weight`
      );
      device.queue.writeBuffer(weightBuffer, 0, projection.weight);
      const weight = createWeightBuffer(
        weightBuffer,
        'f32',
        'row',
        [projection.outputSize, projection.inputSize],
        `embedding_projection_${index}_weight`
      );
      let projected;
      try {
        projected = await runMatmul(
          current,
          weight,
          1,
          projection.outputSize,
          projection.inputSize,
          { transposeB: 'auto', outputDtype: 'f32', kernelPath: config.kernelPath ?? null }
        );
      } finally {
        releaseBuffer(weightBuffer);
      }
      if (current !== input) releaseBuffer(current.buffer);
      current = projected;
      if (projection.bias != null) {
        if (!(projection.bias instanceof Float32Array)
          || projection.bias.length !== projection.outputSize) {
          throw new Error(
            `[Pipeline] Embedding postprocessor projection ${index} bias length mismatch.`
          );
        }
        await runBiasAdd(current, projection.bias, 1, projection.outputSize);
      }
    }
    if (postprocessor.normalize === 'l2') {
      const normalized = await runL2Normalize(current, {
        rowCount: 1,
        hiddenSize: current.shape[1],
      });
      if (current !== input) releaseBuffer(current.buffer);
      current = normalized;
    }
    return current;
  } catch (error) {
    if (current !== input) {
      releaseBuffer(current.buffer);
    }
    throw error;
  }
}

export async function extractEmbeddingFromHiddenGPU(options) {
  const device = getDevice();
  if (!device) throw new Error('[Pipeline] GPU device is required for embedding extraction.');
  const {
    hiddenBuffer,
    activationDtype,
    numTokens,
    hiddenSize,
    embeddingMode,
    finalNorm,
    finalNormBias = null,
    config,
    embeddingPostprocessor = null,
    returnTokenEmbeddings = false,
    sequencePooling = null,
    tokenIds = null,
  } = options;
  if (!finalNorm) throw new Error('[Pipeline] final_norm is required for embedding extraction.');
  const normWeight = resolveFinalNormGpuBuffer(finalNorm, device.queue, 'embedding_final_norm_weight');
  const hidden = createTensor(
    hiddenBuffer,
    activationDtype,
    [numTokens, hiddenSize],
    'embedding_hidden_states'
  );
  let normBiasBuffer = null;
  let normalized = null;
  let pooled = null;
  let sequencePooled = null;
  try {
    if (config.normalizationType === 'layernorm') {
      normBiasBuffer = uploadFinalNormBias(device, finalNormBias, hiddenSize, config);
      normalized = await runLayerNorm(hidden, normWeight.buffer, normBiasBuffer, config.rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        normWeightDtype: requireWeightDtype(finalNorm, 'embedding final LayerNorm weight'),
      });
    } else {
      normalized = await runRMSNorm(hidden, normWeight.buffer, config.rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
    }
    const postprocessorConfig = config.embeddingPostprocessor ?? null;
    const resolvedMode = postprocessorConfig?.poolingMode ?? embeddingMode;
    if (postprocessorConfig && embeddingMode !== resolvedMode) {
      throw new Error(
        `[Pipeline] embeddingMode "${embeddingMode}" conflicts with manifest poolingMode="${resolvedMode}".`
      );
    }
    pooled = await runEmbeddingPool(normalized, {
      rowCount: numTokens,
      hiddenSize,
      mode: resolvedMode,
    });
    const postprocessed = await applyEmbeddingPostprocessor(pooled, embeddingPostprocessor, config);
    if (postprocessed !== pooled) {
      releaseBuffer(pooled.buffer);
      pooled = postprocessed;
    }

    let sequenceControl = null;
    if (sequencePooling) {
      if (!tokenIds || tokenIds.length !== numTokens) {
        throw new Error('[Pipeline] sequence pooling requires the exact input token IDs.');
      }
      sequenceControl = buildSequencePoolControl(tokenIds, sequencePooling);
      sequencePooled = await runEmbeddingPool(normalized, {
        rowCount: numTokens,
        hiddenSize,
        mode: sequencePooling.mode,
        mask: sequenceControl.mask,
        includedCount: sequenceControl.includedCount,
        lastIndex: sequenceControl.lastIndex,
      });
    }

    return {
      embedding: await readTensorOutput(pooled),
      tokenEmbeddings: returnTokenEmbeddings ? await readTensorOutput(normalized) : null,
      pooledSequenceEmbedding: sequencePooled ? await readTensorOutput(sequencePooled) : null,
      tokenMask: sequenceControl?.tokenMask ?? null,
      includedTokenCount: sequenceControl?.includedCount ?? 0,
    };
  } finally {
    if (sequencePooled) releaseBuffer(sequencePooled.buffer);
    if (pooled) releaseBuffer(pooled.buffer);
    if (normalized) releaseBuffer(normalized.buffer);
    if (normBiasBuffer) releaseBuffer(normBiasBuffer);
    if (normWeight.owned) releaseBuffer(normWeight.buffer);
  }
}
