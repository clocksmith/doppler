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

export function resolveDirectF16KVCacheWrite(options) {
  const {
    state,
    layerIdx,
    currentSeqLen,
    numTokens,
    numKVHeads,
    headDim,
    reusesSharedKV,
    storeSharedKV,
    diffusionGemmaDecoder,
    valueNorm,
    diagnosticsEnabled,
    disableRoPE,
  } = options;
  if (
    numTokens !== 1
    || reusesSharedKV === true
    || storeSharedKV === true
    || diffusionGemmaDecoder === true
    || valueNorm === true
    || diagnosticsEnabled === true
    || disableRoPE === true
  ) {
    return null;
  }
  const cache = state?.kvCache;
  if (
    !cache?.hasGPUCache?.()
    || cache.kvDtype !== 'f16'
    || cache.layout !== 'contiguous'
    || cache.windowSize != null
    || cache.layerSpecs != null
    || typeof cache.recordF16UpdateAlreadyWrittenFromGPU !== 'function'
  ) {
    return null;
  }
  const gpuBuffers = cache.getGPUBuffers(layerIdx);
  const layout = gpuBuffers?.layout ?? cache.layout;
  if (layout !== 'contiguous' || !gpuBuffers?.keysGPU || !gpuBuffers?.valuesGPU) {
    return null;
  }
  return {
    keysBuffer: gpuBuffers.keysGPU,
    valuesBuffer: gpuBuffers.valuesGPU,
    dstOffset: currentSeqLen * numKVHeads * headDim,
  };
}

export async function runRecordedAttentionCore(options) {
  const {
    attentionKernelVariant,
    attentionPlan,
    attentionPrecisionContract,
    attentionKernelPath,
    assertDtypeTransition,
    attnScale,
    attnSoftcap,
    cachedKTensor,
    cachedKDtype,
    cachedVTensor,
    cachedVDtype,
    causalForAttention,
    debugFlags,
    decoderKVState,
    effectiveSlidingWindow,
    headDim,
    kTensor,
    kvState,
    layerIdx,
    multimodalBidirectionalSpan,
    numHeads,
    numKVHeads,
    numTokens,
    prefillFallbackNeedsCast,
    qGateTensor,
    qTensor,
    recorder,
    resourceScope,
    skipKVCacheWrites,
    state,
    vTensor,
  } = options;
  let attnOutput = null;
  let attnForProjection = null;

  let attentionOutputGateFused = false;

  const attentionKernelRunners = {
    bdpa: async () => {
      const basisKDtype = 'f16';
      const basisVDtype = 'f16';
      const numBasisVectors = Math.max(1, kvState.bdpaBasisCount);
      const basisKTensor = createTensor(kvState.bdpaBasisK, basisKDtype, [numBasisVectors, numKVHeads * headDim], 'bdpa_basis_k');
      const basisVTensor = createTensor(kvState.bdpaBasisV, basisVDtype, [numBasisVectors, numKVHeads * headDim], 'bdpa_basis_v');

      let qForBDPA = qTensor;
      if (qForBDPA.dtype !== 'f16') {
        assertDtypeTransition(state, qForBDPA.dtype, 'f16', 'BDPA attention would narrow Q implicitly.');
        qForBDPA = await recordCastF32ToF16(recorder, qTensor);
        resourceScope.release(qForBDPA.buffer);
      }

      return recordAttentionBDPA(recorder, qForBDPA, basisKTensor, basisVTensor, kvState.bdpaPagedK, kvState.bdpaPagedV, kvState.bdpaIndex, numHeads, headDim, {
        seqLen: numTokens,
        kvLen: kvState.kvLenForAttention,
        numKVHeads,
        causal: causalForAttention,
        startPos: kvState.startPosForMask,
        layerIdx,
        slidingWindow: effectiveSlidingWindow,
        attnSoftcap,
        scale: attnScale,
        ropeCos: state.ropeFreqsCos,
        ropeSin: state.ropeFreqsSin,
      });
    },
    tieredQuant: async () => {
      let qForAttention = qTensor;
      if (kvState.coldQuantMode !== 'none' && qTensor.dtype !== 'f32') {
        assertDtypeTransition(state, qTensor.dtype, 'f32', 'Tiered quant attention would widen Q implicitly.');
        qForAttention = await recordCastF16ToF32(recorder, qTensor);
        resourceScope.release(qForAttention.buffer);
      }
      if (kvState.coldQuantMode === 'none') {
        throw new Error('Tiered quant attention requires cold quant mode.');
      }
      if (!kvState.coldScalesK || !kvState.coldScalesV) {
        throw new Error('Tiered quant attention requires cold scale buffers.');
      }

      const cachedHotKTensor = createTensor(kvState.cachedKHot, cachedKDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_K_hot');
      const cachedHotVTensor = createTensor(kvState.cachedVHot, cachedVDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_V_hot');
      return recordAttentionTieredQuant(
        recorder,
        qForAttention,
        cachedHotKTensor,
        cachedHotVTensor,
        kvState.cachedKCold,
        kvState.cachedVCold,
        kvState.coldScalesK,
        kvState.coldScalesV,
        numHeads,
        headDim,
        buildTieredQuantAttentionOptions(kvState, {
          seqLen: numTokens,
          numKVHeads,
          causal: causalForAttention,
          startPos: kvState.startPosForMask,
          slidingWindow: effectiveSlidingWindow ?? 0,
          attnSoftcap,
          scale: attnScale,
        })
      );
    },
    contiguousQuant: async () => {
      let qForAttention = qTensor;
      if (qTensor.dtype !== 'f32') {
        assertDtypeTransition(state, qTensor.dtype, 'f32', 'Contiguous quant attention would widen Q implicitly.');
        qForAttention = await recordCastF16ToF32(recorder, qTensor);
        resourceScope.release(qForAttention.buffer);
      }

      if (!kvState.coldScalesK || !kvState.coldScalesV) {
        throw new Error('Contiguous quant attention requires scale buffers.');
      }
      if (!kvState.rotationMatrixBuffer || !kvState.codebookCentroidsBuffer) {
        throw new Error('Contiguous quant attention requires TurboQuant shared buffers.');
      }

      return recordAttentionContiguousQuant(
        recorder,
        qForAttention,
        kvState.cachedKCold,
        kvState.cachedVCold,
        kvState.coldScalesK,
        kvState.coldScalesV,
        numHeads,
        headDim,
        buildContiguousQuantAttentionOptions(kvState, {
          seqLen: numTokens,
          kvLen: kvState.kvLenForAttention,
          numKVHeads,
          causal: causalForAttention,
          startPos: kvState.startPosForMask,
          slidingWindow: effectiveSlidingWindow ?? 0,
          attnSoftcap,
          scale: attnScale,
        })
      );
    },
    tiered: async () => {
      const cachedHotKTensor = createTensor(kvState.cachedKHot, cachedKDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_K_hot');
      const cachedHotVTensor = createTensor(kvState.cachedVHot, cachedVDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_V_hot');
      const cachedColdKTensor = createTensor(kvState.cachedKCold, cachedKDtype, [kvState.coldLen, numKVHeads * headDim], 'cached_K_cold');
      const cachedColdVTensor = createTensor(kvState.cachedVCold, cachedVDtype, [kvState.coldLen, numKVHeads * headDim], 'cached_V_cold');
      return recordAttentionTiered(recorder, qTensor, cachedHotKTensor, cachedHotVTensor, cachedColdKTensor, cachedColdVTensor, numHeads, headDim, {
        seqLen: numTokens,
        coldLen: kvState.coldLen,
        hotLen: kvState.hotLen,
        numKVHeads,
        causal: causalForAttention,
        startPos: kvState.startPosForMask,
        slidingWindow: effectiveSlidingWindow ?? 0,
        attnSoftcap,
        scale: attnScale,
        hotWindow: kvState.hotWindow,
        hotStart: kvState.hotStart,
        coldPageTable: kvState.coldPageTable,
        coldPageSize: kvState.coldPageSize,
        coldLayout: kvState.coldPageTable ? 2 : 0,
        hotLayout: kvState.hotWindow > 0 ? 1 : 0,
      });
    },
    contiguous: async () => {
      // Prefill fallback: quantized/tiered layouts use raw K/V for prefill, cast to f16 to match kernel path
      let kForAttn = cachedKTensor;
      let vForAttn = cachedVTensor;
      if (prefillFallbackNeedsCast) {
        const hasExplicitF16KvContract = isAttentionKvDtypeExplicit(attentionPrecisionContract, 'f16');
        if (cachedKDtype === 'f16' && kTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
          assertDtypeTransition(state, kTensor.dtype, 'f16', 'Prefill fallback attention would narrow K implicitly.');
        }
        if (cachedVDtype === 'f16' && vTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
          assertDtypeTransition(state, vTensor.dtype, 'f16', 'Prefill fallback attention would narrow V implicitly.');
        }
        const kCasted = cachedKDtype === 'f16' && kTensor.dtype !== 'f16'
          ? await recordCastF32ToF16(recorder, kTensor) : kTensor;
        const vCasted = cachedVDtype === 'f16' && vTensor.dtype !== 'f16'
          ? await recordCastF32ToF16(recorder, vTensor) : vTensor;
        kForAttn = createTensor(kCasted.buffer, kCasted.dtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_K');
        vForAttn = createTensor(vCasted.buffer, vCasted.dtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_V');
        if (kTensor.dtype !== 'f16') resourceScope.release(kCasted.buffer);
        if (vTensor.dtype !== 'f16') resourceScope.release(vCasted.buffer);
      }
      // Session precedence was resolved once at pipeline construction. The
      // immutable runtime session is the sole attention policy input.
      // Kernel enforces head_dim=256, f16 KV, contiguous layout; only applies
      // when numTokens > 1 (prefill). The semantic plan owns the flag.
      const useFlashPrefillRec = attentionPlan.attention.flashPrefill === true;
      const useOrtFlashPrefillRec = attentionPlan.attention.ortFlashPrefill === true;
      const useOutputGateFusion = attentionPlan.outputGate.mode === 'attention-epilogue';
      const result = await recordAttention(recorder, qTensor, kForAttn, vForAttn, null, numHeads, headDim, {
        seqLen: numTokens,
        kvLen: kvState.kvLenForAttention,
        numKVHeads,
        causal: causalForAttention,
        bidirectionalSpanStart: multimodalBidirectionalSpan?.start ?? 0,
        bidirectionalSpanLength: multimodalBidirectionalSpan?.length ?? 0,
        startPos: kvState.startPosForMask,
        layerIdx,
        slidingWindow: effectiveSlidingWindow,
        attnSoftcap,
        scale: attnScale,
        kvStart: kvState.kvStart,
        kvLayout: kvState.kvLayout,
        kvPageTable: kvState.kvPageTable,
        kvPageSize: kvState.kvPageSize,
        kernelPath: attentionKernelPath,
        useFlashPrefill: skipKVCacheWrites ? false : useFlashPrefillRec,
        useOrtFlashPrefill: skipKVCacheWrites ? false : useOrtFlashPrefillRec,
        outputGate: useOutputGateFusion ? qGateTensor : null,
      });
      attentionOutputGateFused = result?.outputGateFused === true;
      return result;
    },
  };
  const runAttentionKernel = attentionKernelRunners[attentionPlan.attention.implementation];
  if (!runAttentionKernel) {
    throw new Error(`Unsupported attention kernel variant "${attentionKernelVariant}" at layer ${layerIdx}`);
  }

  try {
    const boundPlan = bindAttentionPlan(attentionPlan, { runAttentionKernel });
    attnOutput = await executeBoundAttentionPlan(
      boundPlan,
      createAttentionExecutor('recorded', {
        attention: (bound) => bound.resources.runAttentionKernel(),
      })
    );
    resourceScope.register(attnOutput.buffer, 'attention-core-output', 'submitOwned');
  } finally {
    if (decoderKVState?.ownedBuffers) {
      for (const buffer of decoderKVState.ownedBuffers) {
        resourceScope.release(buffer);
      }
      decoderKVState.ownedBuffers = null;
    }
  }
  await runProbes('attn_core_out', attnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numHeads * headDim,
    probes: state.debugProbes,
    recorder,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: attnOutput.dtype,
  });
  if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.attn_core_out_HEALTH`,
      attnOutput,
      attnOutput.dtype,
      numTokens * numHeads * headDim
    );
  }

  attnForProjection = attnOutput;
  if (qGateTensor && !attentionOutputGateFused) {
    // The shared plan selects the same gate semantics for both executors.
    // sigmoid(gate) even when the HF config surfaces `output_gate_type=swish`.
    const gateActivation = attentionPlan.outputGate.semantics;
    attnForProjection = await recordSiLU(recorder, attnOutput, {
      size: numTokens * numHeads * headDim,
      gate: qGateTensor,
      useVec4: (numTokens * numHeads * headDim) % 4 === 0,
      gateActivation,
      inputActivation: 'identity',
      swigluLimit: null,
    });
    resourceScope.register(attnForProjection.buffer, 'gated-attention-output', 'submitOwned');
    resourceScope.release(attnOutput.buffer);
  }

  return { attnOutput, attnForProjection };
}

export function finalizeRecordedAttentionSuccess(options) {
  const {
    attentionPlan,
    buffersToTrack,
    finalOutput,
    kTensor,
    qGateTensor,
    qTensor,
    resourceScope,
    retainSharedKvBuffers,
    state,
    vTensor,
  } = options;

  // Track intermediate buffers for cleanup after submit (not release!)
  // These buffers are used by recorded operations that haven't executed yet.
  // Releasing them back to the pool would allow reuse before the encoder is submitted,
  // causing data corruption (especially for small decode buffers).
  if (qTensor) {
    resourceScope.release(qTensor.buffer);
  }
  if (qGateTensor) {
    resourceScope.release(qGateTensor.buffer);
  }
  if (!retainSharedKvBuffers) {
    if (kTensor) {
      resourceScope.release(kTensor.buffer);
    }
    if (vTensor?.buffer && vTensor.buffer !== kTensor?.buffer) {
      resourceScope.release(vTensor.buffer);
    }
  }
  for (const buffer of buffersToTrack) {
    resourceScope.release(buffer);
  }
  resourceScope.register(finalOutput.buffer, 'attention-output', 'submitOwned');
  resourceScope.retain(finalOutput.buffer, 'attention-output', 'returned-to-layer');

  const resourceEvents = resourceScope.close('success');
  if (state.observationContext?.receiptPolicy !== 'off') {
    state.stats.attentionResourceEvents ??= [];
    state.stats.attentionResourceEvents.push(...resourceEvents);
  }
  captureAttentionRefactorReceipt({
    state,
    plan: attentionPlan,
    resourceEvents,
  });
}
