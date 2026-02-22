

// Re-export CPU functions
export { rmsNormCPU, matmulCPU, applySoftcapping, f16ToF32, f16BufferToF32 } from './cpu.js';

// Re-export GPU functions
export { computeLogitsGPU, recordLogitsGPU, computeChunkedLogitsGPU, resolveCpuWeightDims, resolveLmHeadChunkRows, extractLmHeadChunk, writeChunkLogits } from './gpu.js';

// Re-export utilities
export { extractLastPositionLogits, finalizeLogits } from './utils.js';

// Imports for computeLogits orchestrator
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice } from '../../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm, castF16ToF32 } from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { runProbes } from '../probes.js';
import { rmsNormCPU, matmulCPU, f16BufferToF32 } from './cpu.js';
import { resolveCpuWeightDims, computeChunkedLogitsGPU } from './gpu.js';
import { finalizeLogits } from './utils.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';

function shouldForceStableF32Logits(config, inputDtype) {
  // Small Gemma-family checkpoints can overflow in pure F16 logits path after RMSNorm offset.
  return inputDtype === 'f16'
    && config.rmsNormWeightOffset === true
    && Number.isFinite(config.hiddenSize)
    && config.hiddenSize <= 768;
}


export async function computeLogits(
  hiddenStates,
  numTokens,
  weights,
  config,
  useGPU,
  debugFlags = {},
  getNormWeightBuffer,
  debugCheckBuffer,
  debugProbes,
  options = null
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
    activationDtype: activationDtypeOverride,
  } = config;
  const activationDtype = activationDtypeOverride ?? getRuntimeConfig().inference.compute.activationDtype;
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
  
  let cpuWeightVocabSize = null;
  
  let cpuWeightLayout = null;

  if (isCpuWeightBuffer(lmHead)) {
    const dims = resolveCpuWeightDims(lmHead);
    cpuWeightVocabSize = dims.vocabSize;
    cpuWeightLayout = lmHead.layout;
    if (!cpuWeightLayout) {
      throw new Error('LM head CPU weight is missing layout metadata.');
    }
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
    
    let cpuHiddenStates;
    if (inputIsGPU) {
      const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
      const data = await readBuffer(hiddenStates, numTokens * hiddenSize * bytesPerElement);
      const decodeDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: activationDtype });
      cpuHiddenStates = decodeDtype === 'f16'
        ? f16BufferToF32(data)
        : new Float32Array(data);
    } else {
      cpuHiddenStates =  (hiddenStates);
    }
    const normed = rmsNormCPU(
      cpuHiddenStates,
      (finalNorm),
      rmsNormEps,
      config.rmsNormWeightOffset
    );
    const rawLogits = isCpuWeightBuffer(lmHead)
      ? matmulCPU(
        normed,
        lmHead.data,
        numTokens,
        matmulVocabSize,
        hiddenSize,
        cpuWeightLayout,
        cpuWeightLayout === 'column' ? cpuWeightVocabSize : null
      )
      : matmulCPU(normed, (lmHead), numTokens, matmulVocabSize, hiddenSize);
    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // GPU path
  // 1. Get or create input buffer
  
  let inputBuffer;
  let inputBufferOwned = false;
  if (inputIsGPU) {
    inputBuffer =  (hiddenStates);
  } else {
    inputBuffer = acquireBuffer((hiddenStates).byteLength, undefined, 'logits_input');
    device.queue.writeBuffer(inputBuffer, 0, (hiddenStates));
    inputBufferOwned = true;
  }
  const inputDtype = inputIsGPU ? activationDtype : 'f32';
  await runProbes('pre_final_norm', inputBuffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
    dtype: inputDtype,
  });

  // 2. Apply final RMSNorm
  
  let normWeightBuffer;
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else if (finalNorm instanceof GPUBuffer) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer((finalNorm).byteLength, undefined, 'final_norm_w');
    device.queue.writeBuffer(normWeightBuffer, 0, (finalNorm));
  }

  // Debug: Check hidden state before final norm
  if (!debugFlags.finalNormDebugDone && debugCheckBuffer) {
    debugFlags.finalNormDebugDone = true;
    await debugCheckBuffer(inputBuffer, 'Before final norm', numTokens, hiddenSize);
    await debugCheckBuffer(normWeightBuffer, 'Final norm weights', 1, hiddenSize);
  }

  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const forceStableF32Logits = shouldForceStableF32Logits(config, inputDtype);
  let normInputTensor = inputTensor;
  let normInputBufferOwned = false;
  if (forceStableF32Logits) {
    normInputTensor = await castF16ToF32(inputTensor);
    normInputBufferOwned = true;
  }
  const normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });
  if (normInputBufferOwned) {
    releaseBuffer(normInputTensor.buffer);
  }
  await runProbes('final_norm', normedTensor.buffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
    dtype: normedTensor.dtype,
  });

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, normedTensor.buffer, [numTokens, hiddenSize]);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(normedTensor.buffer, 'After final norm', numTokens, hiddenSize);
  }

  if (isCpuWeightBuffer(lmHead)) {
    if (cpuWeightVocabSize == null) {
      throw new Error('LM head CPU weight is missing vocabSize metadata.');
    }
    const rawLogits = await computeChunkedLogitsGPU(
      normedTensor,
      lmHead,
      numTokens,
      hiddenSize,
      matmulVocabSize,
      cpuWeightVocabSize,
      debugProbes,
      largeWeights,
      config.kernelPath ?? null
    );

    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(normedTensor.buffer);
    if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);

    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
  }

  // 3. Project to vocab via LM head
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (lmHead instanceof GPUBuffer) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer((lmHead).byteLength, undefined, 'lm_head_w');
    device.queue.writeBuffer(rawBuffer, 0, (lmHead));
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
    role: forceStableF32Logits ? undefined : 'lm_head',
    kernelPath: config.kernelPath ?? null,
  });
  await runProbes('logits', logitsTensor.buffer, {
    numTokens,
    hiddenSize: matmulVocabSize,
    probes: debugProbes,
    dtype: logitsTensor.dtype,
  });

  // Trace lm_head output
  if (kernelTrace.enabled) {
    await traceStep('matmul', 'lm_head', -1, logitsTensor.buffer, [numTokens, matmulVocabSize]);
  }

  // 4. Read back logits
  const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsTensor.dtype });
  const lastPositionOnly = options?.lastPositionOnly === true && numTokens > 1;
  const logitsReadSize = lastPositionOnly
    ? matmulVocabSize * logitsBytes
    : numTokens * matmulVocabSize * logitsBytes;
  const logitsReadOffset = lastPositionOnly
    ? (numTokens - 1) * matmulVocabSize * logitsBytes
    : 0;
  const logitsData = lastPositionOnly
    ? await readBufferSlice(logitsTensor.buffer, logitsReadOffset, logitsReadSize)
    : await readBuffer(logitsTensor.buffer, logitsReadSize);

  // Cleanup
  if (inputBufferOwned) releaseBuffer(inputBuffer);
  releaseBuffer(normedTensor.buffer);
  releaseBuffer(logitsTensor.buffer);
  if (!getNormWeightBuffer && !(finalNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuffer);
  if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);

  const rawLogits = logitsTensor.dtype === 'f16'
    ? f16BufferToF32(logitsData)
    : new Float32Array(logitsData);
  if (lastPositionOnly) {
    return finalizeLogits(rawLogits, 1, matmulVocabSize, vocabSize, config, debugProbes);
  }
  return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes);
}
