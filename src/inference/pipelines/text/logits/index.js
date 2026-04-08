

// Re-export CPU functions
export { rmsNormCPU, matmulCPU, applySoftcapping, f16ToF32, f16BufferToF32 } from './cpu.js';

// Re-export GPU functions
export { computeLogitsGPU, recordLogitsGPU, computeChunkedLogitsGPU, resolveCpuWeightDims, resolveLmHeadChunkRows, extractLmHeadChunk, writeChunkLogits } from './gpu.js';

// Re-export utilities
export { extractLastPositionLogits, finalizeLogits, readBufferWithCleanup } from './utils.js';

// Imports for computeLogits orchestrator
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm, castF16ToF32, castF32ToF16 } from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { runProbes } from '../probes.js';
import { rmsNormCPU, matmulCPU, f16BufferToF32 } from './cpu.js';
import { resolveCpuWeightDims, computeChunkedLogitsGPU } from './gpu.js';
import { finalizeLogits, readBufferWithCleanup } from './utils.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { getKernelPathMatmulPrecision, getKernelPathStepPrecision } from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';

function shouldForceStableF32Logits(config, inputDtype) {
  if (inputDtype !== 'f16') {
    return false;
  }
  // Softcapped output heads are numerically sensitive in pure F16 on the
  // final RMSNorm + LM-head path. Widen only the logits tail so the main
  // layer stack and KV cache can stay on the faster F16 lane.
  if (Number.isFinite(config.finalLogitSoftcapping) && config.finalLogitSoftcapping > 0) {
    return true;
  }
  // Small Gemma-family checkpoints can also overflow in pure F16 logits path
  // after RMSNorm offset even without output softcapping.
  return config.rmsNormWeightOffset === true
    && Number.isFinite(config.hiddenSize)
    && config.hiddenSize <= 768;
}

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

const STABLE_F32_LOGITS_KERNEL_MAP = new Map([
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
]);

function createStableF32LogitsKernelPath(kernelPath) {
  if (!kernelPath?.postLayer) {
    return kernelPath;
  }
  let changed = false;
  const postLayer = kernelPath.postLayer.map((step) => {
    if (step?.op === 'final_norm') {
      const precision = {
        ...(step.precision ?? {}),
        inputDtype: 'f32',
        outputDtype: 'f32',
      };
      if (
        step.precision?.inputDtype === precision.inputDtype
        && step.precision?.outputDtype === precision.outputDtype
      ) {
        return step;
      }
      changed = true;
      return {
        ...step,
        precision,
      };
    }
    if (step?.op !== 'lm_head' && step?.op !== 'lm_head_prefill') {
      return step;
    }
    const replacement = STABLE_F32_LOGITS_KERNEL_MAP.get(step.kernel) ?? step.kernel;
    const precision = {
      ...(step.precision ?? {}),
      inputDtype: 'f32',
      outputDtype: 'f32',
    };
    if (
      replacement === step.kernel
      && step.precision?.inputDtype === precision.inputDtype
      && step.precision?.outputDtype === precision.outputDtype
    ) {
      return step;
    }
    changed = true;
    return {
      ...step,
      kernel: replacement,
      precision,
    };
  });
  if (!changed) {
    return kernelPath;
  }
  return {
    ...kernelPath,
    postLayer,
  };
}

export function resolveLmHeadMatmulConfig(numTokens, options = null) {
  const lastPositionOnly = options?.lastPositionOnly === true && numTokens > 1;
  return {
    lastPositionOnly,
    matmulRows: lastPositionOnly ? 1 : numTokens,
    phaseOverride: lastPositionOnly ? 'decode' : null,
  };
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
  const inputIsGPU = isGpuBufferInstance(hiddenStates);

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
  if (getNormWeightBuffer) {
    normWeightBuffer = getNormWeightBuffer(finalNorm, 'final_norm_w');
  } else if (isGpuBufferInstance(finalNorm)) {
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

  // Trace final norm output
  if (kernelTrace.enabled) {
    await traceStep('rmsnorm', 'final_norm', -1, finalNormTensor.buffer, [numTokens, hiddenSize]);
  }

  // Debug: Check hidden state after final norm
  if (!debugFlags.afterFinalNormDebugDone && debugCheckBuffer) {
    debugFlags.afterFinalNormDebugDone = true;
    await debugCheckBuffer(finalNormTensor.buffer, 'After final norm', numTokens, hiddenSize);
  }

  if (isCpuWeightBuffer(lmHead)) {
    if (cpuWeightVocabSize == null) {
      throw new Error('LM head CPU weight is missing vocabSize metadata.');
    }
    const rawLogits = await computeChunkedLogitsGPU(
      finalNormTensor,
      lmHead,
      numTokens,
      hiddenSize,
      matmulVocabSize,
      cpuWeightVocabSize,
      debugProbes,
      operatorDiagnostics,
      largeWeights,
      stableKernelPath,
      config.executionPolicies ?? null
    );

    if (inputBufferOwned) releaseBuffer(inputBuffer);
    releaseBuffer(finalNormTensor.buffer);
    if (!getNormWeightBuffer && !isGpuBufferInstance(finalNorm)) releaseBuffer(normWeightBuffer);

    return finalizeLogits(rawLogits, numTokens, matmulVocabSize, vocabSize, config, debugProbes, operatorDiagnostics);
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

  const lastTokenMatmul = resolveLmHeadMatmulConfig(numTokens, options);
  const { lastPositionOnly, matmulRows } = lastTokenMatmul;
  const matmulPhaseOverride = lastTokenMatmul.phaseOverride;
  const lmHeadPhase = matmulPhaseOverride ?? (matmulRows === 1 ? 'decode' : 'prefill');

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
  const lmHeadInputDtype = forceStableF32Logits
    ? matmulInputTensor.dtype
    : resolveMatmulStepDtype('lm_head', lmHeadPhase, stableKernelPath, matmulInputTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = forceStableF32Logits
    ? matmulInputTensor.dtype
    : resolveMatmulStepDtype('lm_head', lmHeadPhase, stableKernelPath, matmulInputTensor.dtype, 'outputDtype');
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

  // HuggingFace models store lm_head as [vocabSize, hiddenSize], so transposeB=true
  const logitsTensor = await runMatmul(matmulInputTensor, lmHeadBuffer, matmulRows, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
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
    if (!getNormWeightBuffer && !isGpuBufferInstance(finalNorm)) releaseBuffer(normWeightBuffer);
    if (lmHeadBufferOwned) releaseBuffer(lmHeadGPU);
  });

  const rawLogits = logitsTensor.dtype === 'f16'
    ? f16BufferToF32(logitsData)
    : new Float32Array(logitsData);
  return finalizeLogits(rawLogits, matmulRows, matmulVocabSize, vocabSize, config, debugProbes, operatorDiagnostics);
}
