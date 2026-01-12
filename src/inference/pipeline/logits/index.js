/**
 * Logits computation - final layer norm and LM head projection.
 *
 * This module handles the final steps of inference:
 * - Apply final RMS norm to hidden states
 * - Project to vocabulary size via LM head
 * - Handle tied embeddings (transposeB for HuggingFace format)
 * - CPU fallback for non-GPU execution
 *
 * @module inference/pipeline/logits
 */

// Re-export CPU functions
export { rmsNormCPU, matmulCPU, applySoftcapping, f16ToF32, f16BufferToF32 } from './cpu.js';

// Re-export GPU functions
export { computeLogitsGPU, recordLogitsGPU, computeChunkedLogitsGPU, resolveCpuWeightDims, resolveLmHeadChunkRows, extractLmHeadChunk, writeChunkLogits } from './gpu.js';

// Re-export utilities
export { extractLastPositionLogits, finalizeLogits } from './utils.js';

// Imports for computeLogits orchestrator
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../gpu/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../../gpu/kernel-selector.js';
import { createTensor } from '../../../gpu/tensor.js';
import { isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import { runProbes } from '../probes.js';
import { rmsNormCPU, matmulCPU, f16BufferToF32 } from './cpu.js';
import { resolveCpuWeightDims, computeChunkedLogitsGPU } from './gpu.js';
import { finalizeLogits } from './utils.js';

/**
 * Compute logits from hidden states.
 *
 * This function:
 * 1. Applies final RMS normalization
 * 2. Projects to vocabulary via LM head matrix multiplication
 * 3. Handles tied embeddings (uses transposeB for HF format)
 * 4. Falls back to CPU if GPU unavailable
 *
 * @param {GPUBuffer | Float32Array} hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param {number} numTokens - Number of tokens (required for GPU buffer input)
 * @param {import('./types.js').LogitsWeights} weights - Final norm and LM head weights
 * @param {import('./types.js').LogitsConfig} config - Model configuration for logits
 * @param {boolean} useGPU - Whether to use GPU
 * @param {import('./types.js').LogitsDebugFlags} [debugFlags={}] - Debug flags to prevent repeated logging
 * @param {(weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer} [getNormWeightBuffer] - Helper to get norm weight buffer (from pipeline)
 * @param {(buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>} [debugCheckBuffer] - Helper for debug buffer checking (from pipeline)
 * @param {import('../../../config/schema/index.js').ProbeConfigSchema[] | null} [debugProbes] - Debug probes configuration
 * @returns {Promise<Float32Array>} Logits tensor [numTokens, vocabSize]
 */
export async function computeLogits(
  hiddenStates,
  numTokens,
  weights,
  config,
  useGPU,
  debugFlags = {},
  getNormWeightBuffer,
  debugCheckBuffer,
  debugProbes
) {
  if (isTraceEnabled('logits')) {
    trace.logits(`LOGITS_ENTRY: numTokens=${numTokens}, useGPU=${useGPU}`);
  }
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    largeWeights,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded, returning zeros');
    return new Float32Array(vocabSize);
  }

  const requestedVocabSize = useTiedEmbeddings && embeddingVocabSize
    ? embeddingVocabSize
    : vocabSize;
  let matmulVocabSize = requestedVocabSize;
  /** @type {number | null} */
  let cpuWeightVocabSize = null;
  /** @type {'row' | 'column' | null} */
  let cpuWeightLayout = null;

  if (isCpuWeightBuffer(lmHead)) {
    const dims = resolveCpuWeightDims(lmHead);
    cpuWeightVocabSize = dims.vocabSize;
    cpuWeightLayout = lmHead.layout;
    if (dims.hiddenSize !== hiddenSize) {
      log.warn('Logits', `LM head hiddenSize mismatch: weight=${dims.hiddenSize}, expected=${hiddenSize}`);
    }
    if (matmulVocabSize > dims.vocabSize) {
      log.warn('Logits', `LM head vocabSize smaller than requested: weight=${dims.vocabSize}, requested=${matmulVocabSize}. Clamping.`);
      matmulVocabSize = dims.vocabSize;
    }
  }

  // Check if input is GPU buffer
  const inputIsGPU = hiddenStates instanceof GPUBuffer;

  // CPU fallback path
  if (isTraceEnabled('logits')) {
    trace.logits(`LOGITS_PATH: device=${!!device}, useGPU=${useGPU}, taking ${(!device || !useGPU) ? 'CPU' : 'GPU'} path`);
  }
  if (!device || !useGPU) {
    /** @type {Float32Array} */
    let cpuHiddenStates;
    if (inputIsGPU) {
      const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
      const data = await readBuffer(hiddenStates, numTokens * hiddenSize * bytesPerElement);
      cpuHiddenStates = activationDtype === 'f16'
        ? f16BufferToF32(data)
        : new Float32Array(data);
    } else {
      cpuHiddenStates = /** @type {Float32Array} */ (hiddenStates);
    }
    const normed = rmsNormCPU(cpuHiddenStates, /** @type {Float32Array} */(finalNorm), rmsNormEps);
    const rawLogits = isCpuWeightBuffer(lmHead)
      ? matmulCPU(
        normed,
        lmHead.data,
        numTokens,
        matmulVocabSize,
        hiddenSize,
        cpuWeightLayout ?? 'row',
        cpuWeightLayout === 'column' ? cpuWeightVocabSize : null
      )
      : matmulCPU(normed, /** @type {Float32Array} */(lmHead), numTokens, matmulVocabSize, hiddenSize);
    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // GPU path
  // 1. Get or create input buffer
  /** @type {GPUBuffer} */
  let inputBuffer;
  let inputBufferOwned = false;
  if (inputIsGPU) {
    inputBuffer = /** @type {GPUBuffer} */ (hiddenStates);
  } else {
    inputBuffer = acquireBuffer(/** @type {Float32Array} */(hiddenStates).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0, /** @type {BufferSource} */(hiddenStates));
    inputBufferOwned = true;
  }
  await runProbes('pre_final_norm', inputBuffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
  });

  /** @type {'f16' | 'f32'} */
  const inputDtype = inputIsGPU ? activationDtype : 'f32';

  // 2. Apply final RMSNorm
  /** @type {GPUBuffer} */
  let normWeightBuffer;
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer(/** @type {Float32Array} */(finalNorm).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0, /** @type {BufferSource} */(finalNorm));
  }

  // Debug: Check hidden state before final norm
  if (!debugFlags.finalNormDebugDone && debugCheckBuffer) {
    debugFlags.finalNormDebugDone = true;
    await debugCheckBuffer(inputBuffer, 'Before final norm', numTokens);
    await debugCheckBuffer(normWeightBuffer, 'Final norm weights', 1, 100);
  }

  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normedTensor = await runRMSNorm(inputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });
  await runProbes('final_norm', normedTensor.buffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
  });

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, normedTensor.buffer, [numTokens, hiddenSize]);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(normedTensor.buffer, 'After final norm', numTokens);
  }

  if (isCpuWeightBuffer(lmHead)) {
    const rawLogits = await computeChunkedLogitsGPU(
      normedTensor,
      lmHead,
      numTokens,
      hiddenSize,
      matmulVocabSize,
      cpuWeightVocabSize ?? matmulVocabSize,
      debugProbes,
      largeWeights
    );

    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(normedTensor.buffer);
    if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);

    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // 3. Project to vocab via LM head
  /** @type {GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer} */
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer(/** @type {Float32Array} */(lmHead).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0, /** @type {BufferSource} */(lmHead));
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  // Debug: Log buffer info for lm_head matmul
  const lmHeadGPU = isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer;
  const lmHeadDtype = getWeightDtype(lmHeadBuffer);  // dtype from WeightBuffer metadata
  const normedDtype = normedTensor.dtype;
  if (isTraceEnabled('logits')) {
    trace.logits(`LM_HEAD_MATMUL: M=${numTokens}, N=${matmulVocabSize}, K=${hiddenSize}, lmHeadDtype=${lmHeadDtype}, normedDtype=${normedDtype}, size=${lmHeadGPU.size}, bufLabel=${lmHeadGPU.label}`);
  }

  // HuggingFace models store lm_head as [vocabSize, hiddenSize], so transposeB=true
  const logitsTensor = await runMatmul(normedTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
  });
  await runProbes('logits', logitsTensor.buffer, {
    numTokens,
    hiddenSize: matmulVocabSize,
    probes: debugProbes,
  });

  // Trace lm_head output
  if (kernelTrace.enabled) {
    await traceStep('matmul', 'lm_head', -1, logitsTensor.buffer, [numTokens, matmulVocabSize]);
  }

  // 4. Read back logits
  const logitsBytes = logitsTensor.dtype === 'f16' ? 2 : 4;
  const logitsData = await readBuffer(logitsTensor.buffer, numTokens * matmulVocabSize * logitsBytes);

  // Cleanup
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedTensor.buffer);
  releaseBuffer(logitsTensor.buffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);

  const rawLogits = logitsTensor.dtype === 'f16'
    ? f16BufferToF32(logitsData)
    : new Float32Array(logitsData);
  return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
}
