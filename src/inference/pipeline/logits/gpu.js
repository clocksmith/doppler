/**
 * GPU implementations for logits computation.
 *
 * Provides GPU-accelerated implementations for computing logits,
 * including both immediate execution and recorded (batched) variants.
 *
 * @module inference/pipeline/logits/gpu
 */

import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../gpu/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../../gpu/kernel-selector.js';
import { recordMatmul } from '../../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../../gpu/kernels/rmsnorm.js';
import { createTensor } from '../../../gpu/tensor.js';
import { castF32ToF16, castF16ToF32, recordCastF16ToF32 } from '../../../gpu/kernels/cast.js';
import { createWeightBuffer, isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { runProbes } from '../probes.js';

/**
 * Resolve CPU weight buffer dimensions for LM head.
 *
 * @param {import('../../../gpu/weight-buffer.js').CpuWeightBuffer} lmHead
 * @returns {{ vocabSize: number; hiddenSize: number }}
 */
export function resolveCpuWeightDims(lmHead) {
  if (lmHead.shape.length !== 2) {
    throw new Error(`[Logits] CPU LM head shape must be 2D, got [${lmHead.shape.join(', ')}]`);
  }
  if (lmHead.layout === 'column') {
    return { hiddenSize: lmHead.shape[0], vocabSize: lmHead.shape[1] };
  }
  return { vocabSize: lmHead.shape[0], hiddenSize: lmHead.shape[1] };
}

/**
 * Calculate the maximum rows per chunk for LM head matmul.
 *
 * @param {GPUDevice} device
 * @param {number} numTokens
 * @param {number} hiddenSize
 * @param {import('../../../config/schema/index.js').LargeWeightConfigSchema} [config]
 * @returns {number}
 */
export function resolveLmHeadChunkRows(
  device,
  numTokens,
  hiddenSize,
  config
) {
  const resolved = config ?? getRuntimeConfig().inference.largeWeights;
  const safety = Math.min(Math.max(resolved.safetyRatio ?? 0.9, 0.1), 1);
  const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxBytes = Math.floor(maxBinding * safety);

  const maxRowsByWeight = Math.floor(maxBytes / (hiddenSize * 4));
  const maxRowsByOutput = Math.floor(maxBytes / (numTokens * 4));
  const maxRows = Math.min(maxRowsByWeight, maxRowsByOutput);

  if (!Number.isFinite(maxRows) || maxRows <= 0) {
    throw new Error(
      `[Logits] LM head chunk size underflow (maxBytes=${maxBytes}, hiddenSize=${hiddenSize}, numTokens=${numTokens}).`
    );
  }

  const override = resolved.lmHeadChunkRows ?? null;
  if (override && override > 0) {
    return Math.min(override, maxRows);
  }
  return maxRows;
}

/**
 * Extract a chunk of the LM head weight matrix.
 *
 * @param {Float32Array} data
 * @param {'row' | 'column'} layout
 * @param {number} hiddenSize
 * @param {number} vocabSize
 * @param {number} rowOffset
 * @param {number} rowCount
 * @returns {Float32Array}
 */
export function extractLmHeadChunk(
  data,
  layout,
  hiddenSize,
  vocabSize,
  rowOffset,
  rowCount
) {
  if (layout === 'row') {
    const start = rowOffset * hiddenSize;
    return data.subarray(start, start + rowCount * hiddenSize);
  }

  const chunk = new Float32Array(hiddenSize * rowCount);
  for (let k = 0; k < hiddenSize; k++) {
    const srcOffset = k * vocabSize + rowOffset;
    const dstOffset = k * rowCount;
    chunk.set(data.subarray(srcOffset, srcOffset + rowCount), dstOffset);
  }
  return chunk;
}

/**
 * Write chunk logits to the full logits buffer.
 *
 * @param {Float32Array} target
 * @param {Float32Array} chunk
 * @param {number} numTokens
 * @param {number} vocabSize
 * @param {number} rowOffset
 * @param {number} rowCount
 * @returns {void}
 */
export function writeChunkLogits(
  target,
  chunk,
  numTokens,
  vocabSize,
  rowOffset,
  rowCount
) {
  for (let t = 0; t < numTokens; t++) {
    const srcOffset = t * rowCount;
    const dstOffset = t * vocabSize + rowOffset;
    target.set(chunk.subarray(srcOffset, srcOffset + rowCount), dstOffset);
  }
}

/**
 * Compute logits using chunked GPU matmul for large LM heads.
 *
 * Used when LM head weights are CPU-resident and too large
 * to fit in a single GPU buffer binding.
 *
 * @param {import('../../../gpu/tensor.js').Tensor} normedTensor
 * @param {import('../../../gpu/weight-buffer.js').CpuWeightBuffer} lmHead
 * @param {number} numTokens
 * @param {number} hiddenSize
 * @param {number} vocabSize
 * @param {number} weightVocabSize
 * @param {import('../../../config/schema/index.js').ProbeConfigSchema[] | null} [debugProbes]
 * @param {import('../../../config/schema/index.js').LargeWeightConfigSchema} [largeWeightConfig]
 * @returns {Promise<Float32Array>}
 */
export async function computeChunkedLogitsGPU(
  normedTensor,
  lmHead,
  numTokens,
  hiddenSize,
  vocabSize,
  weightVocabSize,
  debugProbes,
  largeWeightConfig
) {
  const device = getDevice();
  if (!device) {
    throw new Error('[Logits] GPU device not available for chunked LM head.');
  }

  const chunkRows = resolveLmHeadChunkRows(device, numTokens, hiddenSize, largeWeightConfig);
  const caps = getKernelCapabilities();
  const preferF16 = (largeWeightConfig?.preferF16 ?? true) && lmHead.dtype === 'f16' && caps.hasF16;
  const logits = new Float32Array(numTokens * vocabSize);

  if (isTraceEnabled('logits')) {
    trace.logits(`LM_HEAD_CHUNKED: vocab=${vocabSize}, chunkRows=${chunkRows}, layout=${lmHead.layout}, f16=${preferF16}`);
  }

  for (let rowOffset = 0; rowOffset < vocabSize; rowOffset += chunkRows) {
    const rowCount = Math.min(chunkRows, vocabSize - rowOffset);
    const chunkData = extractLmHeadChunk(
      lmHead.data,
      lmHead.layout,
      hiddenSize,
      weightVocabSize,
      rowOffset,
      rowCount
    );

    const f32Buffer = acquireBuffer(chunkData.byteLength, undefined, 'lm_head_chunk_f32');
    device.queue.writeBuffer(
      f32Buffer,
      0,
      chunkData.buffer,
      chunkData.byteOffset,
      chunkData.byteLength
    );

    const chunkShape = lmHead.layout === 'column'
      ? [hiddenSize, rowCount]
      : [rowCount, hiddenSize];

    let weightBuffer = createWeightBuffer(f32Buffer, 'f32', lmHead.layout, chunkShape, 'lm_head_chunk_f32');

    if (preferF16) {
      const f32Tensor = createTensor(f32Buffer, 'f32', chunkShape, 'lm_head_chunk_f32');
      const f16Tensor = await castF32ToF16(f32Tensor);
      releaseBuffer(f32Buffer);
      weightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', lmHead.layout, chunkShape, 'lm_head_chunk_f16');
    }

    const logitsTensor = await runMatmul(normedTensor, weightBuffer, numTokens, rowCount, hiddenSize, {
      transposeB: 'auto',
      role: 'lm_head',
    });

    if (debugProbes?.length) {
      await runProbes('logits', logitsTensor.buffer, {
        numTokens,
        hiddenSize: rowCount,
        probes: debugProbes,
      });
    }

    const chunkLogitsData = await readBuffer(logitsTensor.buffer, numTokens * rowCount * 4);
    const chunkLogits = new Float32Array(chunkLogitsData);
    writeChunkLogits(logits, chunkLogits, numTokens, vocabSize, rowOffset, rowCount);

    releaseBuffer(logitsTensor.buffer);
    releaseBuffer(weightBuffer.buffer);
  }

  return logits;
}

/**
 * Compute logits and return GPU buffer directly (deferred readback).
 *
 * This variant avoids the ~1MB readback per token, enabling GPU-side sampling.
 * Use with runGPUSample or runArgmax to sample directly on GPU.
 *
 * @param {GPUBuffer | Float32Array} hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param {number} numTokens - Number of tokens
 * @param {import('./types.js').LogitsWeights} weights - Final norm and LM head weights
 * @param {import('./types.js').LogitsConfig} config - Model configuration for logits
 * @param {import('./types.js').LogitsDebugFlags} [debugFlags] - Debug flags to prevent repeated logging (optional)
 * @returns {Promise<{ logitsBuffer: GPUBuffer; vocabSize: number } | null>} GPU buffer containing logits [numTokens, vocabSize]
 */
export async function computeLogitsGPU(
  hiddenStates,
  numTokens,
  weights,
  config,
  debugFlags,
) {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!device) {
    return null;
  }

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded');
    return null;
  }
  if (isCpuWeightBuffer(lmHead)) {
    return null;
  }

  // Get or create input buffer
  /** @type {GPUBuffer} */
  let inputBuffer;
  let inputBufferOwned = false;
  if (hiddenStates instanceof GPUBuffer) {
    inputBuffer = hiddenStates;
  } else {
    inputBuffer = acquireBuffer(/** @type {Float32Array} */ (hiddenStates).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0, /** @type {BufferSource} */ (hiddenStates));
    inputBufferOwned = true;
  }

  // Apply final RMSNorm
  /** @type {GPUBuffer} */
  let normWeightBuffer;
  let normWeightBufferOwned = false;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer(/** @type {Float32Array} */ (finalNorm).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0, /** @type {BufferSource} */ (finalNorm));
    normWeightBufferOwned = true;
  }

  /** @type {'f16' | 'f32'} */
  const inputDtype = hiddenStates instanceof GPUBuffer ? activationDtype : 'f32';
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normInputTensor = inputDtype === 'f16' ? await castF16ToF32(inputTensor) : inputTensor;
  const normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });
  if (normInputTensor !== inputTensor) {
    releaseBuffer(normInputTensor.buffer);
  }

  // Project to vocab via LM head
  /** @type {GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer} */
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer(/** @type {Float32Array} */ (lmHead).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0, /** @type {BufferSource} */ (lmHead));
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;

  const logitsTensor = await runMatmul(normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });

  // Cleanup intermediate buffers (but keep logitsBuffer)
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedTensor.buffer);
  if (normWeightBufferOwned) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize };
}

/**
 * Record logits computation (batched, no submit).
 *
 * This variant uses the CommandRecorder to batch logits computation with
 * preceding layer operations, avoiding a GPU sync point.
 *
 * @param {import('../../../gpu/command-recorder.js').CommandRecorder} recorder - CommandRecorder for batched operations
 * @param {GPUBuffer} hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param {number} numTokens - Number of tokens
 * @param {import('./types.js').LogitsWeights} weights - Final norm and LM head weights
 * @param {import('./types.js').LogitsConfig} config - Model configuration for logits
 * @returns {Promise<{ logitsBuffer: GPUBuffer; vocabSize: number }>} GPU buffer containing logits [numTokens, vocabSize] and vocab size
 */
export async function recordLogitsGPU(
  recorder,
  hiddenStates,
  numTokens,
  weights,
  config,
) {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize ? embeddingVocabSize : vocabSize;

  if (!finalNorm || !lmHead) {
    throw new Error('[recordLogitsGPU] Final norm or LM head not loaded');
  }
  if (isCpuWeightBuffer(lmHead)) {
    throw new Error('[recordLogitsGPU] CPU-resident LM head not supported in recorded path');
  }

  // Get norm weight buffer
  /** @type {GPUBuffer} */
  let normWeightBuffer;
  let normWeightOwned = false;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer(/** @type {Float32Array} */ (finalNorm).byteLength, undefined, 'final_norm_w');
    recorder.device.queue.writeBuffer(normWeightBuffer, 0, /** @type {BufferSource} */ (finalNorm));
    normWeightOwned = true;
  }

  /** @type {'f16' | 'f32'} */
  const inputDtype = activationDtype;
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(hiddenStates, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normInputTensor = inputDtype === 'f16' ? await recordCastF16ToF32(recorder, inputTensor) : inputTensor;

  // Record RMSNorm (no submit)
  const normedTensor = await recordRMSNorm(recorder, normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
  });
  if (normInputTensor !== inputTensor) {
    recorder.trackTemporaryBuffer(normInputTensor.buffer);
  }

  // Get LM head buffer
  /** @type {GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer} */
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer(/** @type {Float32Array} */ (lmHead).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(rawBuffer, 0, /** @type {BufferSource} */ (lmHead));
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  // Record matmul (no submit)
  const logitsTensor = await recordMatmul(recorder, normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });

  // Track intermediate buffer for cleanup after submit
  recorder.trackTemporaryBuffer(normedTensor.buffer);
  if (normWeightOwned) {
    recorder.trackTemporaryBuffer(normWeightBuffer);
  }
  if (lmHeadBufferOwned) {
    recorder.trackTemporaryBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);
  }

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize };
}
