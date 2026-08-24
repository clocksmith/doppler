import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { recordScale } from '../../../../gpu/kernel-selector.js';
import { recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../../../gpu/kernels/rmsnorm.js';
import { recordLmHeadArgmax } from '../../../../gpu/kernels/lm-head-argmax.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
} from '../../../../gpu/kernels/cast.js';
import {
  isCpuWeightBuffer,
  isGpuBufferInstance,
  isSplitWeightBuffer,
  isWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import {
  getKernelPathMatmulPrecision,
  getKernelPathStepPrecision,
} from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { runProbes } from '../probes.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { resolveLogitInputScale } from './scale-policy.js';
import { finalizeLogitOutputTensor } from './output-transform.js';
import {
  createStableF32LogitsKernelPath,
  shouldForceStableF32Logits,
} from './precision-policy.js';

export function resolvePrecisionFieldDtype(precision, fallback, field) {
  const requested = precision?.[field] ?? fallback;
  if (requested == null) {
    return fallback;
  }
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: requested });
}

export function resolveMatmulStepDtype(role, phase, kernelPath, fallback, field) {
  const precision = getKernelPathMatmulPrecision(role, phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

export function resolvePostLayerStepDtype(op, phase, kernelPath, fallback, field) {
  const precision = getKernelPathStepPrecision(op, 'postLayer', phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

export function resolveLmHeadMatmulRole(phase) {
  return phase === 'prefill' ? 'lm_head_prefill' : 'lm_head';
}

export async function coerceTensorDtype(tensor, targetDtype, recorder = null, options = {}) {
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
    return recorder ? await recordCastF32ToF16(recorder, tensor) : await castF32ToF16(tensor);
  }
  if (tensor.dtype === 'f16' && targetDtype === 'f32') {
    return recorder ? await recordCastF16ToF32(recorder, tensor) : await castF16ToF32(tensor);
  }
  throw new Error(`Unsupported logits matmul dtype coercion: ${tensor.dtype} -> ${targetDtype}`);
}

export function resolveFinalNormGpuBuffer(finalNorm, queue, label) {
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
    queue.writeBuffer(buffer, 0, finalNorm);
    return { buffer, owned: true };
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}

async function recordLogitsTailGPU(
  recorder,
  hiddenStates,
  numTokens,
  weights,
  config,
  operatorDiagnostics = null,
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
  if (isCpuWeightBuffer(lmHead) || isSplitWeightBuffer(lmHead)) {
    throw new Error('[recordLogitsGPU] CPU-resident or split LM head not supported in recorded path');
  }

  // Get norm weight buffer
  
  let normWeightBuffer;
  let normWeightOwned = false;
  const resolvedFinalNorm = resolveFinalNormGpuBuffer(finalNorm, recorder.device.queue, 'final_norm_w');
  normWeightBuffer = resolvedFinalNorm.buffer;
  normWeightOwned = resolvedFinalNorm.owned;

  
  const inputDtype = activationDtype;
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(hiddenStates, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const kernelPath = config.kernelPath ?? null;
  const finalNormPrecision = getKernelPathStepPrecision('final_norm', 'postLayer', phase, 0, kernelPath);
  const hasExplicitFinalNormPrecision = finalNormPrecision?.inputDtype != null || finalNormPrecision?.outputDtype != null;
  await runProbes('pre_final_norm', hiddenStates, {
    numTokens,
    hiddenSize,
    recorder,
    operatorDiagnostics,
    dtype: inputDtype,
  });
  const forceStableF32Logits = !hasExplicitFinalNormPrecision && shouldForceStableF32Logits(config, inputDtype);
  const stableKernelPath = forceStableF32Logits
    ? createStableF32LogitsKernelPath(kernelPath)
    : kernelPath;
  let normInputTensor = inputTensor;
  let normInputOwned = false;
  if (forceStableF32Logits) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies: config.executionPolicies ?? null,
      fromDtype: inputTensor.dtype,
      toDtype: 'f32',
      op: 'logits_final_norm',
      detail: 'Stable logits mode would widen activations implicitly before final RMSNorm.',
    });
    normInputTensor = await recordCastF16ToF32(recorder, inputTensor);
    normInputOwned = true;
  } else {
    const finalNormInputDtype = resolvePostLayerStepDtype('final_norm', phase, stableKernelPath, inputTensor.dtype, 'inputDtype');
    normInputTensor = finalNormInputDtype !== inputTensor.dtype
      ? await coerceTensorDtype(inputTensor, finalNormInputDtype, recorder, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : inputTensor;
    normInputOwned = normInputTensor !== inputTensor;
  }
  // Record RMSNorm (no submit)
  const normedTensor = await recordRMSNorm(recorder, normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
    label: 'final_norm',
  });
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
      ? await coerceTensorDtype(normedTensor, finalNormOutputDtype, recorder, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : normedTensor;
  }
  await runProbes('final_norm', finalNormTensor.buffer, {
    numTokens,
    hiddenSize,
    recorder,
    operatorDiagnostics,
    dtype: finalNormTensor.dtype,
  });
  const logitInputScale = resolveLogitInputScale(config);
  let logitInputTensor = finalNormTensor;
  let logitInputOwned = false;
  if (logitInputScale !== 1) {
    logitInputTensor = await recordScale(recorder, finalNormTensor, logitInputScale, {
      count: numTokens * hiddenSize,
    });
    logitInputOwned = true;
  }
  const lmHeadRole = resolveLmHeadMatmulRole(phase);
  const lmHeadInputDtype = forceStableF32Logits
    ? logitInputTensor.dtype
    : resolveMatmulStepDtype(lmHeadRole, phase, stableKernelPath, logitInputTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = forceStableF32Logits
    ? logitInputTensor.dtype
    : resolveMatmulStepDtype(lmHeadRole, phase, stableKernelPath, logitInputTensor.dtype, 'outputDtype');
  const lmHeadInputTensor = lmHeadInputDtype !== logitInputTensor.dtype
    ? await coerceTensorDtype(logitInputTensor, lmHeadInputDtype, recorder, {
      executionPolicies: config.executionPolicies ?? null,
      op: 'lm_head',
      transitionDeclaredBy: 'step_precision',
    })
    : logitInputTensor;

  // Get LM head buffer
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (isGpuBufferInstance(lmHead)) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer( (lmHead).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(rawBuffer, 0,  (lmHead));
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  return {
    hiddenSize,
    matmulVocabSize,
    phase,
    stableKernelPath,
    lmHeadRole,
    lmHeadOutputDtype,
    normedTensor,
    finalNormTensor,
    logitInputTensor,
    logitInputOwned,
    normInputTensor,
    normInputOwned,
    normWeightBuffer,
    normWeightOwned,
    lmHeadInputTensor,
    lmHeadBuffer,
    lmHeadBufferOwned,
  };
}

function trackRecordedLogitsTail(recorder, tail) {
  const trackedTempBuffers = new Set();
  const trackTempBufferOnce = (buffer) => {
    if (!buffer || trackedTempBuffers.has(buffer)) {
      return;
    }
    trackedTempBuffers.add(buffer);
    recorder.trackTemporaryBuffer(buffer);
  };
  if (tail.finalNormTensor !== tail.normedTensor) {
    trackTempBufferOnce(tail.normedTensor.buffer);
  }
  trackTempBufferOnce(tail.finalNormTensor.buffer);
  if (tail.logitInputOwned) {
    trackTempBufferOnce(tail.logitInputTensor.buffer);
  }
  if (tail.lmHeadInputTensor !== tail.logitInputTensor) {
    trackTempBufferOnce(tail.lmHeadInputTensor.buffer);
  }
  if (tail.normWeightOwned) {
    recorder.trackTemporaryBuffer(tail.normWeightBuffer);
  }
  if (tail.normInputOwned) {
    recorder.trackTemporaryBuffer(tail.normInputTensor.buffer);
  }
  if (tail.lmHeadBufferOwned) {
    recorder.trackTemporaryBuffer(isWeightBuffer(tail.lmHeadBuffer) ? tail.lmHeadBuffer.buffer : tail.lmHeadBuffer);
  }
}

export async function recordLogitsGPU(
  recorder,
  hiddenStates,
  numTokens,
  weights,
  config,
  operatorDiagnostics = null,
) {
  const tail = await recordLogitsTailGPU(
    recorder,
    hiddenStates,
    numTokens,
    weights,
    config,
    operatorDiagnostics
  );

  let logitsTensor = await recordMatmul(recorder, tail.lmHeadInputTensor, tail.lmHeadBuffer, numTokens, tail.matmulVocabSize, tail.hiddenSize, {
    transposeB: 'auto',
    role: tail.lmHeadRole,
    kernelPath: tail.stableKernelPath,
    outputDtype: tail.lmHeadOutputDtype,
    executionPolicies: config.executionPolicies ?? null,
  });
  logitsTensor = await finalizeLogitOutputTensor(logitsTensor, config, {
    recorder, numTokens, vocabSize: tail.matmulVocabSize, operatorDiagnostics,
  });

  trackRecordedLogitsTail(recorder, tail);

  return { logitsBuffer: logitsTensor.buffer, vocabSize: tail.matmulVocabSize, logitsDtype: logitsTensor.dtype };
}

export async function recordGreedyLmHeadArgmaxGPU(
  recorder,
  hiddenStates,
  numTokens,
  weights,
  config,
  options,
  operatorDiagnostics = null,
) {
  if (numTokens !== 1) {
    throw new Error(`[recordGreedyLmHeadArgmaxGPU] expected numTokens=1, got ${numTokens}.`);
  }
  const tail = await recordLogitsTailGPU(
    recorder,
    hiddenStates,
    numTokens,
    weights,
    config,
    operatorDiagnostics
  );
  const outputBuffer = await recordLmHeadArgmax(recorder, tail.lmHeadInputTensor, tail.lmHeadBuffer, {
    vocabSize: tail.matmulVocabSize,
    hiddenSize: tail.hiddenSize,
    padTokenId: options.padTokenId,
    logitSoftcap: options.logitSoftcap,
    outputBuffer: options.outputBuffer,
    outputIndex: options.outputIndex,
  });
  trackRecordedLogitsTail(recorder, tail);
  return outputBuffer;
}

