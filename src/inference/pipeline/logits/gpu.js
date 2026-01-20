

import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../../gpu/kernel-selector.js';
import { recordMatmul } from '../../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../../gpu/kernels/rmsnorm.js';
import { createTensor } from '../../../gpu/tensor.js';
import { castF32ToF16 } from '../../../gpu/kernels/cast.js';
import { createWeightBuffer, isWeightBuffer, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { runProbes } from '../probes.js';
import { f16BufferToF32 } from './cpu.js';


export function resolveCpuWeightDims(lmHead) {
  if (lmHead.shape.length !== 2) {
    throw new Error(`[Logits] CPU LM head shape must be 2D, got [${lmHead.shape.join(', ')}]`);
  }
  if (lmHead.layout === 'column') {
    return { hiddenSize: lmHead.shape[0], vocabSize: lmHead.shape[1] };
  }
  return { vocabSize: lmHead.shape[0], hiddenSize: lmHead.shape[1] };
}


export function resolveLmHeadChunkRows(
  device,
  numTokens,
  hiddenSize,
  config
) {
  const resolved = config ?? getRuntimeConfig().inference.largeWeights;
  if (resolved.safetyRatio == null) {
    throw new Error('runtime.inference.largeWeights.safetyRatio is required.');
  }
  const safety = Math.min(Math.max(resolved.safetyRatio, 0.1), 1);
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
  if (!largeWeightConfig) {
    throw new Error('[Logits] largeWeights config is required for chunked LM head.');
  }

  const chunkRows = resolveLmHeadChunkRows(device, numTokens, hiddenSize, largeWeightConfig);
  const caps = getKernelCapabilities();
  const weightDtype = selectRuleValue('inference', 'dtype', 'lmHeadChunkWeightDtype', {
    preferF16: largeWeightConfig.preferF16,
    lmHeadDtype: lmHead.dtype,
    hasF16: caps.hasF16,
  });
  const preferF16 = weightDtype === 'f16';
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
        dtype: logitsTensor.dtype,
      });
    }

    const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsTensor.dtype });
    const chunkLogitsData = await readBuffer(logitsTensor.buffer, numTokens * rowCount * logitsBytes);
    const chunkLogits = logitsTensor.dtype === 'f16'
      ? f16BufferToF32(chunkLogitsData)
      : new Float32Array(chunkLogitsData);
    writeChunkLogits(logits, chunkLogits, numTokens, vocabSize, rowOffset, rowCount);

    releaseBuffer(logitsTensor.buffer);
    releaseBuffer(weightBuffer.buffer);
  }

  return logits;
}


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
    activationDtype,
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!device) {
    return null;
  }
  if (!activationDtype) {
    throw new Error('[Logits] activationDtype is required.');
  }

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded');
    return null;
  }
  if (isCpuWeightBuffer(lmHead)) {
    return null;
  }

  // Get or create input buffer
  
  let inputBuffer;
  let inputBufferOwned = false;
  if (hiddenStates instanceof GPUBuffer) {
    inputBuffer = hiddenStates;
  } else {
    inputBuffer = acquireBuffer( (hiddenStates).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0,  (hiddenStates));
    inputBufferOwned = true;
  }

  // Apply final RMSNorm
  
  let normWeightBuffer;
  let normWeightBufferOwned = false;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer( (finalNorm).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0,  (finalNorm));
    normWeightBufferOwned = true;
  }

  
  const inputDtype = hiddenStates instanceof GPUBuffer ? activationDtype : 'f32';
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const normedTensor = await runRMSNorm(inputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });

  // Project to vocab via LM head
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer( (lmHead).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0,  (lmHead));
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

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize, logitsDtype: logitsTensor.dtype };
}


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
  
  let normWeightBuffer;
  let normWeightOwned = false;
  if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer( (finalNorm).byteLength, undefined, 'final_norm_w');
    recorder.device.queue.writeBuffer(normWeightBuffer, 0,  (finalNorm));
    normWeightOwned = true;
  }

  
  const inputDtype = activationDtype;
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(hiddenStates, inputDtype, [numTokens, hiddenSize], 'logits_input');
  // Record RMSNorm (no submit)
  const normedTensor = await recordRMSNorm(recorder, inputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });

  // Get LM head buffer
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer( (lmHead).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(rawBuffer, 0,  (lmHead));
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

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize, logitsDtype: logitsTensor.dtype };
}
