import { isGpuBufferInstance, isWeightBuffer, getWeightDtype } from '../../../../../gpu/weight-buffer.js';
import { getKernelCapabilities } from '../../../../../gpu/device.js';
import { acquireBuffer } from '../../../../../memory/buffer-pool.js';
import {
  recordMatmul,
  recordRMSNorm,
  recordLayerNorm,
  recordBiasAdd,
  recordRoPE,
  canUseRoPEQK,
  recordRoPEQK,
  recordAttention,
  recordAttentionTiered,
  recordAttentionTieredQuant,
  recordAttentionContiguousQuant,
  recordAttentionBDPA,
  recordSiLU,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordMatmulResidualFused,
} from '../../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../../gpu/tensor.js';
import { applyLoRA } from '../../lora-apply.js';
import { getLoRAModule } from '../../lora.js';
import { log, trace } from '../../../../../debug/index.js';
import { selectRuleValue } from '../../../../../rules/rule-registry.js';
import {
  recordAttentionInputs,
  shouldForceF32AttentionProjectionForRoPE,
  resolveAttentionProjectionOutputDtype,
  projectAttentionQKV,
  applyAttentionQKNorm,
  applyAttentionValueNorm,
  hasAttentionProjectionDiagnostics,
  hasAttentionStageDiagnostics,
  resolveAttentionQKNormState,
} from '../projections.js';
import { prepareAttentionProjectionInput } from '../output-projection.js';
import { runProbes } from '../../probes.js';
import { shouldDebugLayer } from '../types.js';
import {
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../../config/kernel-path-loader.js';
import {
  resolveKVCacheState,
  createDiffusionGemmaDecoderKVState,
  buildAttentionDispatchParams,
  buildAttentionInputsData,
} from '../dispatch-params.js';
import {
  buildTieredQuantAttentionOptions,
  buildContiguousQuantAttentionOptions,
} from '../quant-options.js';
import { assertImplicitDtypeTransitionAllowed } from '../../dtype-contract.js';
import {
  resolveAttentionPrecisionContract,
  isAttentionKvDtypeExplicit,
} from '../precision-contract.js';
import { canUseRmsNormWideTileProjectionFusion } from '../rmsnorm-fusion-gate.js';
import { getVectorTensor } from '../../weights.js';
import { resolveAttentionRuntimeSession } from '../../resolved-runtime-session.js';
import {
  bindAttentionPlan,
  createAttentionExecutor,
  executeBoundAttentionPlan,
  resolveAttentionPlanForDispatch,
} from '../plan.js';
import { createRecordedResourceScope } from '../../../../resource-scope.js';
import { captureAttentionRefactorReceipt } from '../receipt.js';
import { applyAttentionQueryScale } from '../query-transform.js';
import { resolveQueryScale } from '../heterogeneous-contract.js';
import { enqueueRecordedTensorHealth, shouldTraceRecordedHealth } from '../recorded-health.js';
import { resolveDirectF16KVCacheWrite, runRecordedAttentionCore } from './recorded.js';

export function assertAttentionDtypeTransitionAllowed(state, fromDtype, toDtype, detail, transitionDeclaredBy = null) {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: state?.executionPolicies ?? null,
    fromDtype,
    toDtype,
    op: 'attention',
    detail,
    transitionDeclaredBy,
  });
}

export function resolveRecordedAttentionDispatch(options) {
  const {
    attentionActivationDtype,
    attentionPrecisionContract,
    canFuseInputNormProjRec,
    config,
    diffusionGemmaDecoder,
    headDim,
    hiddenSize,
    input,
    isPrefill,
    kTensor,
    kvCacheDtype,
    kvState,
    layerIdx,
    layerType,
    layerWeights,
    lora,
    matmulOutputDtype,
    normed,
    numHeads,
    numKVHeads,
    numTokens,
    oProjInputDtype,
    oProjOutputDtype,
    qGateTensor,
    qTensor,
    qkNormApplied,
    queryPreAttnScalar,
    reusesSharedKV,
    ropeApplied,
    runtimeSession,
    kernelPath,
    skipKVCacheWrites,
    slidingWindow,
    state,
    storeSharedKV,
    usedFusedQKV,
    useF16Activations,
    vTensor,
  } = options;
  const dispatchConfig = {
    layerIdx, numTokens, isPrefill, numHeads, numKVHeads, headDim, hiddenSize,
    slidingWindow: diffusionGemmaDecoder ? null : slidingWindow,
    layerType, layerTypes: config.layerTypes,
    queryPreAttnScalar,
    causalAttention: diffusionGemmaDecoder ? false : config.causalAttention,
    activationDtype: attentionActivationDtype,
    kvCacheDtype: attentionPrecisionContract.resolvedKvCacheDtype ?? state.kvCache?.kvDtype ?? null,
  };
  const dispatchParams = buildAttentionDispatchParams(dispatchConfig, state, kTensor, vTensor, kvState);
  const {
    effectiveSlidingWindow, attentionKernelVariant, attnScale,
    cachedKDtype, cachedVDtype, cachedKTensor, cachedVTensor,
    prefillFallbackNeedsCast, causalForAttention,
  } = dispatchParams;
  const attentionKernelPath = skipKVCacheWrites ? null : kernelPath;

  // 5. Attention

  recordAttentionInputs(state, buildAttentionInputsData(
    dispatchConfig, input, normed, kvState, dispatchParams,
    { useF16Activations, matmulOutputDtype },
    usedFusedQKV, qTensor, kTensor, vTensor,
  ));
  const mergedSessionRec = runtimeSession;
  const plannedOProjLoRA = getLoRAModule(lora, layerIdx, 'o_proj');
  const attentionPlan = resolveAttentionPlanForDispatch({
    config: {
      ...config,
      kernelPath: attentionKernelPath,
    },
    inputDtype: input.dtype,
    activationDtype: attentionActivationDtype,
    projectionDtype: matmulOutputDtype,
    kvDtype: kvCacheDtype,
    outputProjectionInputDtype: oProjInputDtype,
    outputDtype: oProjOutputDtype,
    outputProjectionWeightDtype: layerWeights.oProj
      ? getWeightDtype(layerWeights.oProj)
      : null,
    outputProjectionHasLoRA: plannedOProjLoRA != null,
    session: mergedSessionRec,
    capabilities: getKernelCapabilities(),
    fusion: {
      inputNormProjection: canFuseInputNormProjRec,
      qkvProjection: usedFusedQKV,
      qkNorm: qkNormApplied,
      qkNormRoPE: qkNormApplied && ropeApplied,
    },
    qGateTensor,
    diagnosticsEligible: hasAttentionProjectionDiagnostics(state)
      || hasAttentionStageDiagnostics(state, ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']),
    sharedKV: {
      reuses: reusesSharedKV,
      stores: storeSharedKV,
    },
    kvState,
    dispatchParams,
  });
  if (attentionPlan.attention.implementation !== attentionKernelVariant) {
    throw new Error(
      `Attention plan/dispatch mismatch at layer ${layerIdx}: ` +
      `plan=${attentionPlan.attention.implementation}, dispatch=${attentionKernelVariant}.`
    );
  }
  if (state.observationContext?.receiptPolicy !== 'off') {
    state.stats.attentionPlans ??= [];
    state.stats.attentionPlans.push(attentionPlan);
  }

  return {
    attentionKernelVariant,
    attentionKernelPath,
    attentionPlan,
    attnScale,
    cachedKTensor,
    cachedKDtype,
    cachedVTensor,
    cachedVDtype,
    causalForAttention,
    effectiveSlidingWindow,
    prefillFallbackNeedsCast,
  };
}
