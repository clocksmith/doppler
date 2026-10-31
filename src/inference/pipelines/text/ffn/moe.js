import { resolveGroupedPrefill } from '../moe/grouped-prefill.js';


import { releaseOrTrack } from '../ops.js';
import { runQwenSharedExpert, resolveQwenSharedSlot } from './qwen-shared-expert.js';
import { createTensor } from '../../../../gpu/tensor.js';


export async function runMoEFFNGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  options = {}
) {
  const { config, moeRouter, expertWeights, expertLoader, layerRouterWeights } = context;
  const groupedPrefill = resolveGroupedPrefill({ kernelPath: context.kernelPath ?? null, phase: context.phase, layerIdx });
  if (groupedPrefill && (config.expertFormat !== 'qwen3_5'
    || context.runtimeMoERoutingConfig?.activeExpertSelection !== 'topk-route')) {
    throw new Error('[MoE grouped prefill] Requires the declared GPU topk-route executor.');
  }
  const sharedSlot = resolveQwenSharedSlot({ kernelPath: context.kernelPath ?? null, phase: context.phase, layerIdx });
  if (sharedSlot && (config.expertFormat !== 'qwen3_5'
    || context.runtimeMoERoutingConfig?.activeExpertSelection !== 'topk-route')) {
    throw new Error('[MoE] Shared slot requires the declared Qwen topk-route executor.');
  }

  if (!moeRouter || !expertWeights || !expertLoader) {
    throw new Error('MoE components not initialized');
  }
  if (!Number.isFinite(config.numExperts) || config.numExperts <= 0) {
    throw new Error('MoE config is missing numExperts.');
  }
  if (!Number.isFinite(config.moeTopK) || config.moeTopK <= 0) {
    throw new Error('MoE config is missing moeTopK.');
  }

  const { moeFeedForwardGPU } = await import('../moe-impl.js');
  if (typeof config.modelType !== 'string' || config.modelType.length === 0) {
    throw new Error('MoE config.modelType is required; re-convert the model with an explicit manifest modelType.');
  }

  const outputBuffer = await moeFeedForwardGPU(
    inputTensor.buffer,
    numTokens,
    {
      modelType: config.modelType,
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      rmsNormEps: config.rmsNormEps,
      expertIntermediateSize: config.moeExpertIntermediateSize,
      numExperts: config.numExperts,
      moeTopK: config.moeTopK,
      expertFormat: config.expertFormat,
      hiddenActivation: config.hiddenActivation,
      swigluLimit: config.swigluLimit,
      activationDtype: inputTensor.dtype,
      activeExpertSelection: context.runtimeMoERoutingConfig?.activeExpertSelection,
      routerInputBuffer: options.routerInputTensor?.buffer ?? null,
      routerInputDtype: options.routerInputTensor?.dtype ?? null,
      kernelPath: context.kernelPath ?? null,
      executionPolicies: context.executionPolicies ?? null,
      groupedPrefill,
      sharedSlot: sharedSlot ? { weights: context.weights.get(`layer_${layerIdx}`),
        intermediateSize: config.moeSharedExpertIntermediateSize, kernelPath: context.kernelPath,
        phase: context.phase } : null,
    },
    moeRouter,
    expertWeights,
    expertLoader,
    layerIdx,
     (layerRouterWeights),
    context.recorder ?? null
  );

  const routed = createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');
  if (sharedSlot || config.expertFormat !== 'qwen3_5') return routed;
  try {
    return await runQwenSharedExpert(inputTensor, routed, context.weights.get(`layer_${layerIdx}`), {
      numTokens, hiddenSize: config.hiddenSize, intermediateSize: config.moeSharedExpertIntermediateSize,
      kernelPath: context.kernelPath ?? null, swigluLimit: config.swigluLimit,
      recorder: context.recorder ?? null,
      layerIdx,
      phase: context.phase,
    });
  } finally {
    releaseOrTrack(context.recorder, routed.buffer);
  }
}
