import { resolveGroupedPrefill } from '../moe/grouped-prefill.js';


import { releaseOrTrack } from '../ops.js';
import { runQwenSharedExpert, resolveDeclaredTopK, resolveQwenNormRouter, resolveQwenSharedSlot, resolveQwenSharedSlotVariants } from './qwen-shared-expert.js';
import { createTensor } from '../../../../gpu/tensor.js';

// Lazily loaded once: each dynamic import() makes the browser re-walk the module graph.
let moeImplModule = null;


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
  // A residual-bearing shared-slot down adds the residual processFFNStandard staged;
  // the layer then skips its own add, so the residual is added exactly once.
  const residualFused = sharedSlot && resolveQwenSharedSlotVariants({
    kernelPath: context.kernelPath, phase: context.phase, layerIdx,
  }).down === 'pair_combine_residual_f32';
  const pendingResidual = residualFused ? (context.__pendingFfnResidualTensor ?? null) : null;
  // A declared moe_norm_router runs the post-attention norm processFFNStandard handed over
  // together with the router projections; it writes the normalized row into inputTensor.
  const normRouter = resolveQwenNormRouter({ kernelPath: context.kernelPath ?? null, phase: context.phase, layerIdx });
  const pendingNorm = normRouter ? (context.__pendingFfnNorm ?? null) : null;
  if (normRouter && (!sharedSlot || !pendingNorm)) {
    throw new Error(`[MoE] Layer ${layerIdx} declares moe_norm_router, which needs the shared-slot route and the `
      + 'post-attention norm from processFFNStandard (sandwich norms and layer plans run their own norm).');
  }
  if (residualFused && !pendingResidual) {
    throw new Error(`[MoE] Layer ${layerIdx} declares a residual-bearing shared-slot down, but its caller `
      + 'did not hand over the FFN residual (sandwich norms, layer plans and residual branch scales keep their own add).');
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

  const { moeFeedForwardGPU } = await (moeImplModule ??= import('../moe-impl.js'));
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
      phase: context.phase,
      signal: context.signal,
      activity: context.operatorDiagnostics?.activity ?? null,
      sharedSlot: sharedSlot ? { weights: context.weights.get(`layer_${layerIdx}`),
        intermediateSize: config.moeSharedExpertIntermediateSize, kernelPath: context.kernelPath,
        phase: context.phase,
        residual: pendingResidual
          ? { tensor: pendingResidual, outputBuffer: context.__pendingFfnResidualOutputBuffer ?? null }
          : null } : null,
      topKVariant: resolveDeclaredTopK({ kernelPath: context.kernelPath ?? null, phase: context.phase, layerIdx }),
      normRouter: pendingNorm
        ? { ...pendingNorm, variant: normRouter, sharedRouterWeight: context.weights.get(`layer_${layerIdx}`)?.sharedExpertRouter }
        : null,
    },
    moeRouter,
    expertWeights,
    expertLoader,
    layerIdx,
     (layerRouterWeights),
    context.recorder ?? null
  );

  if (residualFused) context.__ffnResidualFusedFired = true;
  if (pendingNorm) context.__ffnNormFusedFired = true;
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
