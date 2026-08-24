import {
  isCpuWeightBuffer,
  isGpuBufferInstance,
  isSplitWeightBuffer,
  isWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import { resolveActiveExecutionPlan } from '../execution-plan.js';

export function getLogitsWeights(state) {
  const finalNorm = state.weights.get('final_norm');
  const lmHead = state.weights.get('lm_head');
  const lmHeadBias = state.weights.get('lm_head_bias') ?? null;
  const finalNormBias = state.weights.get('final_norm_bias') ?? null;
  if (
    !finalNorm
    || !(isGpuBufferInstance(finalNorm) || finalNorm instanceof Float32Array || isWeightBuffer(finalNorm))
  ) {
    throw new Error('Final norm not found or invalid type');
  }
  if (
    !lmHead
    || !(
      isGpuBufferInstance(lmHead)
      || lmHead instanceof Float32Array
      || isWeightBuffer(lmHead)
      || isCpuWeightBuffer(lmHead)
      || isSplitWeightBuffer(lmHead)
    )
  ) {
    throw new Error('LM head not found or invalid type');
  }
  if (lmHeadBias !== null && !(lmHeadBias instanceof Float32Array)) {
    throw new Error('LM head bias is not a Float32Array');
  }
  if (finalNormBias !== null && !(finalNormBias instanceof Float32Array)) {
    throw new Error('Final norm bias is not a Float32Array');
  }
  return { finalNorm, finalNormBias, lmHead, lmHeadBias };
}

export function getLogitsConfig(state) {
  const config = state.modelConfig;
  const activeExecutionPlan = resolveActiveExecutionPlan(state);
  const activeKernelPath = activeExecutionPlan.kernelPath ?? state.resolvedKernelPath ?? null;
  return {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    normalizationType: config.normalizationType,
    finalNormBiasTensor: config.finalNormBiasTensor,
    rmsNormEps: config.rmsNormEps,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
    useTiedEmbeddings: state.useTiedEmbeddings,
    embeddingVocabSize: state.embeddingVocabSize,
    finalLogitSoftcapping: config.finalLogitSoftcapping,
    logitInputScale: config.logitInputScale,
    logitOutputScale: config.logitOutputScale,
    largeWeights: state.runtimeConfig.inference.largeWeights,
    activationDtype: activeExecutionPlan.activationDtype,
    kernelPath: activeKernelPath,
    executionPolicies: state.executionV1State?.policies ?? null,
    debugProbes: state.runtimeConfig.shared.debug.probes,
  };
}
