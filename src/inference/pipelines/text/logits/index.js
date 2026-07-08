

// Re-export CPU functions
export { rmsNormCPU, matmulCPU, applySoftcapping, f16ToF32, f16BufferToF32 } from './cpu.js';

// Re-export GPU functions
export { computeLogitsGPU, recordLogitsGPU, recordGreedyLmHeadArgmaxGPU, computeChunkedLogitsGPU, resolveCpuWeightDims, resolveLmHeadChunkRows, extractLmHeadChunk, writeChunkLogits } from './gpu.js';

// Re-export utilities
export { extractLastPositionLogits, finalizeLogits, readBufferWithCleanup } from './utils.js';

// Imports for computeLogits orchestrator
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm, runScale, castF16ToF32, castF32ToF16, runLmHeadSelectLogitsF16 } from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { runProbes } from '../probes.js';
import { rmsNormCPU, matmulCPU, f16BufferToF32 } from './cpu.js';
import { resolveCpuWeightDims, computeChunkedLogitsGPU, computeSplitLogitsGPU } from './gpu.js';
import { finalizeLogits, readBufferWithCleanup, resolveLogitInputScale } from './utils.js';
import { getLogitsHealth } from '../debug-utils/index.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { getKernelPathMatmulPrecision, getKernelPathStepPrecision } from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { shouldForceStableF32Logits, createStableF32LogitsKernelPath } from './precision-policy.js';

function resolvePrecisionFieldDtype(precision, fallback, field) {
  const requested = precision?.[field] ?? fallback;
  if (requested == null) {
    return fallback;
  }
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: requested });
}

function resolveMatmulStepDtype(role, phase, kernelPath, fallback, field) {
  const precision = getKernelPathMatmulPrecision(role, phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

function resolvePostLayerStepDtype(op, phase, kernelPath, fallback, field) {
  const precision = getKernelPathStepPrecision(op, 'postLayer', phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

function resolveLmHeadMatmulRole(phase) {
  return phase === 'prefill' ? 'lm_head_prefill' : 'lm_head';
}

function normalizeSelectedLogitTokenIds(value, vocabSize) {
  if (value == null) {
    return null;
  }
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new Error('[Logits] selectedTokenIds must be an array or typed array.');
  }
  const tokenIds = Array.from(value, (entry, index) => {
    const tokenId = Number(entry);
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      throw new Error(
        `[Logits] selectedTokenIds[${index}] must be an integer in [0, ${vocabSize}), got "${String(entry)}".`
      );
    }
    return tokenId;
  });
  if (tokenIds.length === 0) {
    throw new Error('[Logits] selectedTokenIds must not be empty.');
  }
  return tokenIds;
}

async function coerceTensorDtype(tensor, targetDtype, options = {}) {
  if (!targetDtype || tensor.dtype === targetDtype) {
    return tensor;
  }
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: options.executionPolicies ?? null,
    fromDtype: tensor.dtype,
    toDtype: targetDtype,
    op: options.op ?? 'logits',
    detail: 'The execution graph must declare this cast explicitly.',
    transitionDeclaredBy: options.transitionDeclaredBy ?? null,
  });
  if (tensor.dtype === 'f32' && targetDtype === 'f16') {
    return castF32ToF16(tensor);
  }
  if (tensor.dtype === 'f16' && targetDtype === 'f32') {
    return castF16ToF32(tensor);
  }
  throw new Error(`Unsupported logits matmul dtype coercion: ${tensor.dtype} -> ${targetDtype}`);
}

async function traceTensorHealth(label, tensor, elementCount) {
  if (!isTraceEnabled('logits')) {
    return;
  }
  const dtype = tensor?.dtype;
  const buffer = tensor?.buffer;
  if (!buffer || (dtype !== 'f16' && dtype !== 'f32')) {
    return;
  }
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  const data = await readBuffer(buffer, elementCount * bytesPerElement);
  const values = dtype === 'f16' ? f16BufferToF32(data) : new Float32Array(data);
  trace.logits(label, getLogitsHealth(values));
}

export function resolveLmHeadMatmulConfig(numTokens, options = null) {
  const lastPositionOnly = options?.lastPositionOnly === true && numTokens > 1;
  return {
    lastPositionOnly,
    matmulRows: lastPositionOnly ? 1 : numTokens,
    phaseOverride: lastPositionOnly ? 'prefill' : null,
  };
}

function resolveFinalNormGpuBuffer(finalNorm, device, label) {
  if (isWeightBuffer(finalNorm)) {
    return { buffer: finalNorm.buffer, owned: false };
  }
  if (isGpuBufferInstance(finalNorm)) {
    return { buffer: finalNorm, owned: false };
  }
  if (!ArrayBuffer.isView(finalNorm)) {
    throw new Error('[Logits] final_norm must be a GPU buffer, typed array, or WeightBuffer.');
  }
  const buffer = acquireBuffer(finalNorm.byteLength, undefined, label);
  try {
    device.queue.writeBuffer(buffer, 0, finalNorm);
    return { buffer, owned: true };
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
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
  options = null,
  operatorDiagnostics = null
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

  // Consistency check: warn if lmHead weight dtype disagrees with layer activation dtype.
  const lmHeadWeightDtype = getWeightDtype(lmHead);
  if (lmHeadWeightDtype && activationDtype && lmHeadWeightDtype !== activationDtype) {
    log.debug(
      'Logits',
      `Logits lmHead weight dtype "${lmHeadWeightDtype}" differs from layer activationDtype "${activationDtype}". ` +
      'This may be intentional (e.g., F32 weights with F16 activations) but could affect precision.'
    );
  }

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
  let splitWeightVocabSize = null;

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
  if (isSplitWeightBuffer(lmHead)) {
    const dims = resolveCpuWeightDims(lmHead);
    splitWeightVocabSize = dims.vocabSize;
    if (dims.hiddenSize !== hiddenSize) {
      log.warn('Logits', `LM head hiddenSize mismatch: weight=${dims.hiddenSize}, expected=${hiddenSize}`);
    }
    if (matmulVocabSize > dims.vocabSize) {
      log.warn('Logits', `LM head vocabSize smaller than requested: weight=${dims.vocabSize}, requested=${matmulVocabSize}. Clamping.`);
      matmulVocabSize = dims.vocabSize;
    }
  }
  const selectedTokenIds = normalizeSelectedLogitTokenIds(options?.selectedTokenIds, matmulVocabSize);

  // Check if input is GPU buffer
  const inputIsGPU = isGpuBufferInstance(hiddenStates);

  // CPU fallback path
  if (isTraceEnabled('logits')) {
    trace.logits(`LOGITS_PATH: device=${!!device}, useGPU=${useGPU}, taking ${(!device || !useGPU) ? 'CPU' : 'GPU'} path`);
  }
  if (selectedTokenIds && (!device || !useGPU)) {
    throw new Error('[Logits] selectedTokenIds requires the GPU logits path.');
  }
  if (selectedTokenIds && (isCpuWeightBuffer(lmHead) || isSplitWeightBuffer(lmHead))) {
    throw new Error('[Logits] selectedTokenIds requires a single GPU-resident LM head weight buffer.');
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
    const logitInputScale = resolveLogitInputScale(config);
    if (logitInputScale !== 1) {
      for (let i = 0; i < normed.length; i += 1) {
        normed[i] *= logitInputScale;
      }
    }
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
    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes, operatorDiagnostics);
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
    operatorDiagnostics,
    dtype: inputDtype,
  });

  // 2. Apply final RMSNorm
  
  let normWeightBuffer;
  let normWeightBufferOwned = false;
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else {
    const resolvedFinalNorm = resolveFinalNormGpuBuffer(finalNorm, device, 'final_norm_w');
    normWeightBuffer = resolvedFinalNorm.buffer;
    normWeightBufferOwned = resolvedFinalNorm.owned;
  }

  // Debug: Check hidden state before final norm
  if (!debugFlags.finalNormDebugDone && debugCheckBuffer) {
    debugFlags.finalNormDebugDone = true;
    await debugCheckBuffer(inputBuffer, 'Before final norm', numTokens, hiddenSize);
    await debugCheckBuffer(normWeightBuffer, 'Final norm weights', 1, hiddenSize);
  }

  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
  await traceTensorHealth('LOGITS_INPUT_HEALTH', inputTensor, numTokens * hiddenSize);
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const kernelPath = config.kernelPath ?? null;
  const finalNormPrecision = getKernelPathStepPrecision('final_norm', 'postLayer', phase, 0, kernelPath);
  const hasExplicitFinalNormPrecision = finalNormPrecision?.inputDtype != null || finalNormPrecision?.outputDtype != null;
  const forceStableF32Logits = !hasExplicitFinalNormPrecision && shouldForceStableF32Logits(config, inputDtype);
  const stableKernelPath = forceStableF32Logits
    ? createStableF32LogitsKernelPath(kernelPath)
    : kernelPath;
  let normInputTensor = inputTensor;
  let normInputBufferOwned = false;
  if (forceStableF32Logits) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies: config.executionPolicies ?? null,
      fromDtype: inputTensor.dtype,
      toDtype: 'f32',
      op: 'logits_final_norm',
      detail: 'Stable logits mode would widen activations implicitly before final RMSNorm.',
    });
    normInputTensor = await castF16ToF32(inputTensor);
    normInputBufferOwned = true;
  } else {
    const finalNormInputDtype = resolvePostLayerStepDtype('final_norm', phase, stableKernelPath, inputTensor.dtype, 'inputDtype');
    normInputTensor = finalNormInputDtype !== inputTensor.dtype
      ? await coerceTensorDtype(inputTensor, finalNormInputDtype, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : inputTensor;
    normInputBufferOwned = normInputTensor !== inputTensor;
  }
  let normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });
  if (normInputBufferOwned) {
    releaseBuffer(normInputTensor.buffer);
  }
  let finalNormTensor = normedTensor;
  if (!forceStableF32Logits) {
    const finalNormOutputDtype = resolvePostLayerStepDtype(
      'final_norm',
      phase,
      stableKernelPath,
      normedTensor.dtype,
      'outputDtype'
    );
    finalNormTensor = finalNormOutputDtype !== normedTensor.dtype
      ? await coerceTensorDtype(normedTensor, finalNormOutputDtype, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : normedTensor;
  }
  if (finalNormTensor !== normedTensor) {
    releaseBuffer(normedTensor.buffer);
    normedTensor = null;
  }
  await runProbes('final_norm', finalNormTensor.buffer, {
    numTokens,
    hiddenSize,
    probes: debugProbes,
    operatorDiagnostics,
    dtype: finalNormTensor.dtype,
  });
  await traceTensorHealth('FINAL_NORM_HEALTH', finalNormTensor, numTokens * hiddenSize);

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, finalNormTensor.buffer, [numTokens, hiddenSize]);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(finalNormTensor.buffer, 'After final norm', numTokens, hiddenSize);
  }

  const logitInputScale = resolveLogitInputScale(config);
  if (logitInputScale !== 1) {
    const unscaledFinalNormTensor = finalNormTensor;
    finalNormTensor = await runScale(finalNormTensor, logitInputScale, {
      count: numTokens * hiddenSize,
    });
    releaseBuffer(unscaledFinalNormTensor.buffer);
  }

  const lastTokenMatmul = resolveLmHeadMatmulConfig(numTokens, options);
  const { lastPositionOnly, matmulRows } = lastTokenMatmul;
  const matmulPhaseOverride = lastTokenMatmul.phaseOverride;
  const lmHeadPhase = matmulPhaseOverride ?? (matmulRows === 1 ? 'decode' : 'prefill');
  const lmHeadRole = resolveLmHeadMatmulRole(lmHeadPhase);

  let matmulInputTensor = finalNormTensor;
  let matmulInputOwned = false;
  if (lastPositionOnly) {
    const inputBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: finalNormTensor.dtype });
    const rowSize = hiddenSize * inputBytes;
    const rowOffset = (numTokens - 1) * rowSize;
    const lastInputBuffer = acquireBuffer(rowSize, undefined, 'logits_input_last');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(finalNormTensor.buffer, rowOffset, lastInputBuffer, 0, rowSize);
    device.queue.submit([encoder.finish()]);
    matmulInputTensor = createTensor(lastInputBuffer, finalNormTensor.dtype, [1, hiddenSize], 'logits_input_last');
    matmulInputOwned = true;
  }

  if (isCpuWeightBuffer(lmHead) || isSplitWeightBuffer(lmHead)) {
    const weightVocabSize = isCpuWeightBuffer(lmHead) ? cpuWeightVocabSize : splitWeightVocabSize;
    if (weightVocabSize == null) {
      throw new Error('LM head weight is missing vocabSize metadata.');
    }
    const rawLogits = isCpuWeightBuffer(lmHead)
      ? await computeChunkedLogitsGPU(
        matmulInputTensor,
        lmHead,
        matmulRows,
        hiddenSize,
        matmulVocabSize,
        weightVocabSize,
        debugProbes,
        operatorDiagnostics,
        largeWeights,
        stableKernelPath,
        config.executionPolicies ?? null
      )
      : await computeSplitLogitsGPU(
        matmulInputTensor,
        lmHead,
        matmulRows,
        hiddenSize,
        matmulVocabSize,
        weightVocabSize,
        debugProbes,
        operatorDiagnostics,
        stableKernelPath,
        config.executionPolicies ?? null
      );

    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(finalNormTensor.buffer);
    if (matmulInputOwned) releaseBuffer(matmulInputTensor.buffer);
    if (normWeightBufferOwned) releaseBuffer(normWeightBuffer);

    return finalizeLogits(rawLogits, matmulRows, matmulVocabSize, vocabSize, config, debugProbes, operatorDiagnostics);
  }

  // 3. Project to vocab via LM head
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (isGpuBufferInstance(lmHead)) {
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
  const lmHeadDtype = getWeightDtype(lmHeadBuffer);
  const normedDtype = finalNormTensor.dtype;
  if (isTraceEnabled('logits')) {
    trace.logits(
      `LM_HEAD_MATMUL: M=${matmulRows}, N=${matmulVocabSize}, K=${hiddenSize}, ` +
      `phase=${matmulPhaseOverride ?? 'auto'}, lmHeadDtype=${lmHeadDtype}, ` +
      `normedDtype=${normedDtype}, size=${lmHeadGPU.size}, bufLabel=${lmHeadGPU.label}`
    );
  }

  const lmHeadInputDtype = forceStableF32Logits
    ? matmulInputTensor.dtype
    : resolveMatmulStepDtype(lmHeadRole, lmHeadPhase, stableKernelPath, matmulInputTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = forceStableF32Logits
    ? matmulInputTensor.dtype
    : resolveMatmulStepDtype(lmHeadRole, lmHeadPhase, stableKernelPath, matmulInputTensor.dtype, 'outputDtype');
  if (lmHeadInputDtype !== matmulInputTensor.dtype) {
    const coercedInput = await coerceTensorDtype(matmulInputTensor, lmHeadInputDtype, {
      executionPolicies: config.executionPolicies ?? null,
      op: 'lm_head',
      transitionDeclaredBy: 'step_precision',
    });
    if (matmulInputOwned) {
      releaseBuffer(matmulInputTensor.buffer);
    }
    matmulInputTensor = coercedInput;
    matmulInputOwned = true;
  }

  if (selectedTokenIds) {
    if (lastPositionOnly !== true) {
      throw new Error('[Logits] selectedTokenIds requires lastPositionOnly=true.');
    }
    const selected = await runLmHeadSelectLogitsF16(matmulInputTensor, lmHeadBuffer, {
      device,
      hiddenSize,
      vocabSize: matmulVocabSize,
      tokenIds: selectedTokenIds,
      hiddenOffset: 0,
      logitSoftcap: config.finalLogitSoftcapping == null ? 0 : Number(config.finalLogitSoftcapping),
    });
    const selectedBytes = selectedTokenIds.length * Float32Array.BYTES_PER_ELEMENT;
    const selectedData = await readBufferWithCleanup(selected.outputBuffer, selectedBytes, () => {
      releaseBuffer(selected.outputBuffer);
      releaseBuffer(selected.tokenIdBuffer);
      if (inputBufferOwned) releaseBuffer(inputBuffer);
      releaseBuffer(finalNormTensor.buffer);
      if (matmulInputOwned) releaseBuffer(matmulInputTensor.buffer);
      if (normWeightBufferOwned) releaseBuffer(normWeightBuffer);
      if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);
    });
    const selectedLogits = new Float32Array(selectedData);
    await runProbes('logits_final', selectedLogits, {
      numTokens: 1,
      hiddenSize: selectedTokenIds.length,
      probes: debugProbes,
      operatorDiagnostics,
    });
    return selectedLogits;
  }

  // HuggingFace models store lm_head as [vocabSize, hiddenSize], so transposeB=true
  const logitsTensor = await runMatmul(matmulInputTensor, lmHeadBuffer, matmulRows, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: lmHeadRole,
    phaseOverride: matmulPhaseOverride,
    kernelPath: stableKernelPath,
    outputDtype: lmHeadOutputDtype,
    executionPolicies: config.executionPolicies ?? null,
  });
  await runProbes('logits', logitsTensor.buffer, {
    numTokens: matmulRows,
    hiddenSize: matmulVocabSize,
    probes: debugProbes,
    operatorDiagnostics,
    dtype: logitsTensor.dtype,
  });

  // Trace lm_head output
  if (kernelTrace.enabled) {
    await traceStep('matmul', 'lm_head', -1, logitsTensor.buffer, [matmulRows, matmulVocabSize]);
  }

  // 4. Read back logits
  const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsTensor.dtype });
  const logitsReadSize = matmulRows * matmulVocabSize * logitsBytes;
  const logitsData = await readBufferWithCleanup(logitsTensor.buffer, logitsReadSize, () => {
    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(finalNormTensor.buffer);
    if (matmulInputOwned) releaseBuffer(matmulInputTensor.buffer);
    releaseBuffer(logitsTensor.buffer);
    if (normWeightBufferOwned) releaseBuffer(normWeightBuffer);
    if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);
  });

  const rawLogits = logitsTensor.dtype === 'f16'
    ? f16BufferToF32(logitsData)
    : new Float32Array(logitsData);
  if (isTraceEnabled('logits')) {
    trace.logits('LM_HEAD_RAW_LOGITS_HEALTH', getLogitsHealth(rawLogits));
  }
  return finalizeLogits(rawLogits, matmulRows, matmulVocabSize, vocabSize, config, debugProbes, operatorDiagnostics);
}
