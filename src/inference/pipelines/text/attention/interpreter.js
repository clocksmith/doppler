

import { isGpuBufferInstance, isWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { getKernelCapabilities } from '../../../../gpu/device.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
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
} from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { log, trace } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
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
} from './projections.js';
import { prepareAttentionProjectionInput } from './output-projection.js';
import { runProbes } from '../probes.js';
import { shouldDebugLayer } from './types.js';
import {
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../config/kernel-path-loader.js';
import {
  resolveKVCacheState,
  createDiffusionGemmaDecoderKVState,
  buildAttentionDispatchParams,
  buildAttentionInputsData,
} from './dispatch-params.js';
import {
  buildTieredQuantAttentionOptions,
  buildContiguousQuantAttentionOptions,
} from './quant-options.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import {
  resolveAttentionPrecisionContract,
  isAttentionKvDtypeExplicit,
} from './precision-contract.js';
import { canUseRmsNormWideTileProjectionFusion } from './rmsnorm-fusion-gate.js';
import { getVectorTensor } from '../weights.js';
import { resolveAttentionRuntimeSession } from '../resolved-runtime-session.js';
import {
  bindAttentionPlan,
  createAttentionExecutor,
  executeBoundAttentionPlan,
  resolveAttentionPlanForDispatch,
} from './plan.js';
import { createRecordedResourceScope } from '../../../resource-scope.js';
import { captureAttentionRefactorReceipt } from './receipt.js';
import { applyAttentionQueryScale } from './query-transform.js';
import { resolveQueryScale } from './heterogeneous-contract.js';
import { enqueueRecordedTensorHealth, shouldTraceRecordedHealth } from './recorded-health.js';

const ATTENTION_DTYPE_LOGGED = new Set();

function resolveDirectF16KVCacheWrite(options) {
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

function assertAttentionDtypeTransitionAllowed(state, fromDtype, toDtype, detail, transitionDeclaredBy = null) {
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: state?.executionPolicies ?? null,
    fromDtype,
    toDtype,
    op: 'attention',
    detail,
    transitionDeclaredBy,
  });
}


export async function interpretAttentionWithRecorder(
  recorder,
  input,
  layerWeights,
  config,
  state,
  debug = false,
  debugFlags = {},
  getWeightBuffer,
  getNormWeightBuffer,
  debugCheckBuffer,
  lora
) {
  const {
    layerIdx,
    numTokens,
    isPrefill,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    rmsNormEps,
    currentSeqLen,
    slidingWindow,
    layerType,
    residualTensor,
    attnSoftcap,
    queryPreAttnScalar,
    queryScale = 1,
    skipInputNorm = false,
    tokenIds = null,
    kernelPath = null,
    disableRoPE = false,
    multimodalBidirectionalSpan = null,
    sharedKVSourceLayerIdx = null,
    storeSharedKV = false,
    diffusionGemmaDecoder = false,
    precomputedInputNorm = null,
  } = config;

  const phase = isPrefill ? 'prefill' : 'decode';
  const resolvedQueryScale = resolveQueryScale(queryScale);
  const runtimeSession = resolveAttentionRuntimeSession(state);
  const resourceScope = createRecordedResourceScope(recorder);
  const attentionPrecisionContract = resolveAttentionPrecisionContract(config, state);
  const attentionActivationDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: attentionPrecisionContract.resolvedActivationDtype ?? config.activationDtype,
  });
  const oProjPrecision = getKernelPathMatmulPrecision('o_proj', phase, layerIdx, kernelPath);
  const oProjInputDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: oProjPrecision?.inputDtype ?? attentionActivationDtype,
  });
  const oProjOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: oProjPrecision?.outputDtype
      ?? attentionPrecisionContract.resolvedOutputDtype
      ?? config.activationDtype,
  });
  const wantsF16Output = oProjOutputDtype === 'f16';
  const useF16Activations = attentionActivationDtype === 'f16';
  const kvCacheFallback = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: useF16Activations });
  const kvCacheDtype = attentionPrecisionContract.resolvedKvCacheDtype ?? state.kvCache?.kvDtype ?? kvCacheFallback;
  const allowF16Attention = useF16Activations && kvCacheDtype === 'f16';
  const attentionInputTransitionDeclaredBy = attentionPrecisionContract.explicitInputDtype
    ? 'step_precision'
    : null;
  let attentionInput = input;
  let attentionInputTemp = false;
  let normed = attentionInput;
  let qTensor = null;
  let qGateTensor = null;
  let kTensor = null;
  let vTensor = null;
  let attnOutput = null;
  let attnForProjection = null;
  let output = null;
  let finalOutput = null;
  let oProjInputTemp = null;
  let retainSharedKvBuffers = false;
  let valueAliasesKey = false;
  let decoderKVState = null;
  let decoderKTemp = null;
  let decoderVTemp = null;
  let attentionPlan = null;
  if (!allowF16Attention && input.dtype !== attentionActivationDtype) {
    assertAttentionDtypeTransitionAllowed(
      state,
      input.dtype,
      attentionActivationDtype,
      'The attention kernel selection would widen the input implicitly.',
      attentionInputTransitionDeclaredBy
    );
    attentionInput = attentionActivationDtype === 'f16'
      ? await recordCastF32ToF16(recorder, input)
      : await recordCastF16ToF32(recorder, input);
    resourceScope.register(attentionInput.buffer, 'attention-input-cast', 'submitOwned');
    attentionInputTemp = true;
    normed = attentionInput;
  }

  if (!layerWeights) {
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const output = createTensor(outputBuf, oProjOutputDtype, [numTokens, hiddenSize], 'attn_output');
    resourceScope.register(output.buffer, 'attention-output', 'retained');
    resourceScope.close('success');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  // Opt-in fusion: when useFusedRmsnormWideTile is set, defer the standalone
  // rmsnorm into each q/k/v_proj matmul (fused kernel runs norm internally).
  let fusedNormWeightRec = null;
  let fusedNormEpsRec = null;
  let fusedNormOffsetRec = false;
  let fusedNormOwnedRec = false;
  const rmsNormFusionFlagRec = runtimeSession.useFusedRmsnormWideTile === true;
  const canSelectFusedRmsNormProjectionRec = canUseRmsNormWideTileProjectionFusion(
    layerWeights,
    sharedKVSourceLayerIdx != null
  );
  const canFuseInputNormProjRec = rmsNormFusionFlagRec
    && config.normalizationType === 'rmsnorm'
    && canSelectFusedRmsNormProjectionRec
    && !skipInputNorm
    && layerWeights.inputNorm
    && getNormWeightBuffer
    && isPrefill
    && numTokens > 1
    && attentionInput.dtype === 'f32';

  try {
  if (precomputedInputNorm) {
    if (
      isPrefill
      || numTokens !== 1
      || skipInputNorm
      || !layerWeights.inputNorm
      || !getNormWeightBuffer
      || precomputedInputNorm.dtype !== attentionInput.dtype
      || precomputedInputNorm.dtype !== 'f32'
    ) {
      throw new Error(`Layer ${layerIdx} received an incompatible precomputed input norm tensor.`);
    }
    normed = precomputedInputNorm;
    await runProbes('post_input_norm', normed.buffer, {
      layerIdx, numTokens, hiddenSize,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: normed.dtype,
    });
  } else if (canFuseInputNormProjRec) {
    fusedNormWeightRec = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    fusedNormEpsRec = rmsNormEps;
    fusedNormOffsetRec = config.rmsNormWeightOffset === true;
    fusedNormOwnedRec = !isGpuBufferInstance(layerWeights.inputNorm) && !isWeightBuffer(layerWeights.inputNorm);
    if (fusedNormOwnedRec) {
      resourceScope.register(fusedNormWeightRec, 'fused-input-norm-weight', 'submitOwned');
    }
    // Keep normed = attentionInput (raw). Each q/k/v_proj matmul runs norm internally.
  } else if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    let normBiasTensor = null;
    let normBiasOwned = false;
    if (config.normalizationType === 'layernorm') {
      if (!layerWeights.inputNormBias) {
        throw new Error(`Layer ${layerIdx} requires input_layernorm.bias for LayerNorm execution.`);
      }
      ({ tensor: normBiasTensor, owned: normBiasOwned } = getVectorTensor(
        layerWeights.inputNormBias,
        `L${layerIdx}.input_norm_bias`,
        hiddenSize,
        {},
        debugFlags
      ));
      normed = await recordLayerNorm(
        recorder,
        attentionInput,
        normWeightBuf,
        normBiasTensor.buffer,
        rmsNormEps,
        { batchSize: numTokens, hiddenSize, label: 'input_norm' }
      );
    } else {
      normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
        label: 'input_norm',
      });
    }
    if (normBiasOwned && normBiasTensor) resourceScope.release(normBiasTensor.buffer);
    if (!isGpuBufferInstance(layerWeights.inputNorm) && !isWeightBuffer(layerWeights.inputNorm)) resourceScope.release(normWeightBuf);
    await runProbes('post_input_norm', normed.buffer, {
      layerIdx, numTokens, hiddenSize,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: normed.dtype,
    });
  }

  const debugLayers = debugFlags.debugLayers;
  const shouldLogLayer = debugLayers === null ? layerIdx === 0 : shouldDebugLayer(layerIdx, debugLayers);
  if (shouldLogLayer) {
    const phase = selectRuleValue('kernels', 'attention', 'phase', { isDecode: !isPrefill });
    const logKey = `L${layerIdx}_${phase}_dtypes`;
    if (!ATTENTION_DTYPE_LOGGED.has(logKey)) {
      ATTENTION_DTYPE_LOGGED.add(logKey);
      trace.attn(layerIdx, `dtypes: activation=${attentionActivationDtype}, input=${input.dtype}, normed=${normed.dtype}`);
    }
  }

  // 2. Q/K/V projections
  const qProjVariant = getKernelPathMatmulVariant('q_proj', phase, layerIdx, kernelPath);
  const kernelPathIsF16 = qProjVariant != null && qProjVariant.includes('f16') && !qProjVariant.includes('f32');
  const matmulOutputDtype = resolveAttentionProjectionOutputDtype(attentionActivationDtype, {
    forceF32: shouldForceF32AttentionProjectionForRoPE({
      attentionInputDtype: attentionActivationDtype,
      headDim,
      rotaryDim: config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      kernelPathIsF16,
    }),
  });
  let usedFusedQKV = false;
  const sharedKVEntry = sharedKVSourceLayerIdx == null
    ? null
    : (state.sharedAttentionState?.get(sharedKVSourceLayerIdx) ?? null);
  if (sharedKVSourceLayerIdx != null && !sharedKVEntry) {
    throw new Error(
      `Layer ${layerIdx} requires shared K/V from layer ${sharedKVSourceLayerIdx}, ` +
      'but no shared K/V state was stored for that source layer.'
    );
  }
  if (sharedKVEntry && (
    sharedKVEntry.headDim !== headDim
      || sharedKVEntry.numKVHeads !== numKVHeads
  )) {
    throw new Error(
      `Layer ${layerIdx} shared K/V geometry mismatch. ` +
      `Expected numKVHeads=${numKVHeads}, headDim=${headDim}; ` +
      `got numKVHeads=${sharedKVEntry.numKVHeads}, headDim=${sharedKVEntry.headDim}.`
    );
  }
  const reusesSharedKV = sharedKVEntry != null;
  retainSharedKvBuffers = reusesSharedKV || storeSharedKV;
  let qkNormApplied = false;
  let ropeApplied = false;
  const qkNormState = resolveAttentionQKNormState({ config, layerWeights, layerIdx, reusesSharedKV });
  const qkNormProjectionDiagnostics = hasAttentionProjectionDiagnostics(state)
    || shouldTraceRecordedHealth(layerIdx, debugFlags);
  const qkNormRoPEDiagnostics = hasAttentionStageDiagnostics(
    state,
    ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']
  )
    || shouldTraceRecordedHealth(layerIdx, debugFlags);
  const qkNormFusionFlag = runtimeSession?.useFusedQKVSplitQKNorm === true;
  const qkNormRoPEFusionFlag = runtimeSession?.useFusedQKVSplitQKNormRoPE === true;
  const skipKVCacheWrites = state.skipKVCacheWrites === true;
  const directF16KVCacheWrite = skipKVCacheWrites
    ? null
    : resolveDirectF16KVCacheWrite({
      state,
      layerIdx,
      currentSeqLen,
      numTokens,
      numKVHeads,
      headDim,
      reusesSharedKV,
      storeSharedKV,
      diffusionGemmaDecoder,
      valueNorm: config.valueNorm === true,
      diagnosticsEnabled: qkNormRoPEDiagnostics,
      disableRoPE,
    });
  let kvCacheWriteFused = false;
  ({ qTensor, qGateTensor, kTensor, vTensor, usedFusedQKV, valueAliasesKey, qkNormApplied, ropeApplied, kvCacheWriteFused } = await projectAttentionQKV({
    recorder,
    normed,
    layerWeights,
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    layerIdx,
    kernelPath,
    matmulOutputDtype,
    getWeightBuffer,
    lora,
    matmulDebug: state.runtimeConfig?.shared?.debug?.matmul ?? null,
    attentionOutputGate: config.attentionOutputGate === true,
    sharedKTensor: sharedKVEntry?.kTensor ?? null,
    sharedVTensor: sharedKVEntry?.vTensor ?? null,
    executionPolicies: state.executionPolicies ?? null,
    releaseTemporary: (buffer) => resourceScope.release(buffer),
    onFusedQKV: layerIdx === 0 && isPrefill
      ? ({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize }) => {
        trace.attn(layerIdx, `Using fused QKV path: ${qSizeFused}+${kSizeFused}+${vSizeFused}=${totalSize}`);
      }
      : null,
    fusedNormWeight: fusedNormWeightRec,
    fusedNormEps: fusedNormEpsRec,
    fusedNormOffset: fusedNormOffsetRec,
    qkNormFusion: {
      enabled: qkNormFusionFlag
        && qkNormState.wantsQKNorm
        && config.queryKeyNormType === 'rmsnorm'
        && config.queryKeyNormAxis === 'head',
      getNormWeightBuffer,
      rmsNormEps,
      rmsNormWeightOffset: qkNormState.rmsNormWeightOffset,
      skipKNorm: qkNormState.skipKNorm,
      allowUnitQKNorm: qkNormState.allowUnitQKNorm,
      projectionDiagnosticsEnabled: qkNormProjectionDiagnostics,
    },
    qkNormRoPEFusion: {
      enabled: qkNormRoPEFusionFlag
        && qkNormState.wantsQKNorm
        && config.queryKeyNormType === 'rmsnorm'
        && config.queryKeyNormAxis === 'head'
        && resolvedQueryScale === 1
        && !disableRoPE
        && !!state.ropeFreqsCos
        && !!state.ropeFreqsSin,
      getNormWeightBuffer,
      rmsNormEps,
      rmsNormWeightOffset: qkNormState.rmsNormWeightOffset,
      skipKNorm: qkNormState.skipKNorm,
      allowUnitQKNorm: qkNormState.allowUnitQKNorm,
      projectionDiagnosticsEnabled: qkNormRoPEDiagnostics,
      freqsCos: state.ropeFreqsCos,
      freqsSin: state.ropeFreqsSin,
      startPos: currentSeqLen,
      rotaryDim: config.ropeRotaryDim,
      pairSpanDim: config.ropeFrequencyBaseDim ?? config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      reusesSharedKV,
      f16KVCacheWrite: directF16KVCacheWrite,
    },
  }));
  if (!kvCacheWriteFused && (!kTensor || !vTensor)) {
    throw new Error('Recorded attention projection returned missing K/V tensors without a fused KV cache write.');
  }
  // Deferred release of the norm weight buffer for fused path.
  if (fusedNormWeightRec && fusedNormOwnedRec) {
    resourceScope.release(fusedNormWeightRec);
  }
  if (qTensor) {
    await runProbes('q_proj', qTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: qTensor.dtype,
    });
  }
  if (kTensor) {
    await runProbes('k_proj', kTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: kTensor.dtype,
    });
  }
  if (vTensor) {
    await runProbes('v_proj', vTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: vTensor.dtype,
    });
  }
  if (qTensor && shouldTraceRecordedHealth(layerIdx, debugFlags)) {
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.q_proj_HEALTH`,
      qTensor,
      qTensor.dtype,
      numTokens * numHeads * headDim
    );
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.k_proj_HEALTH`,
      kTensor,
      kTensor?.dtype ?? null,
      numTokens * numKVHeads * headDim
    );
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.v_proj_HEALTH`,
      vTensor,
      vTensor?.dtype ?? null,
      numTokens * numKVHeads * headDim
    );
  }

  // Optional Q/K normalization. The manifest owns both its numerical family
  // and whether it applies per head or across the complete projection.
  if (qkNormState.wantsQKNorm && !qkNormApplied) {
    ({ qTensor, kTensor } = await applyAttentionQKNorm({
      recorder,
      qTensor,
      kTensor,
      layerWeights,
      getNormWeightBuffer,
      rmsNormEps,
      numTokens,
      numHeads,
      numKVHeads,
      headDim,
      rmsNormWeightOffset: qkNormState.rmsNormWeightOffset,
      normalizationType: config.queryKeyNormType,
      normalizationAxis: config.queryKeyNormAxis,
      releaseTemporary: (buffer) => resourceScope.release(buffer),
      skipKNorm: qkNormState.skipKNorm,
      retainKInput: valueAliasesKey,
      allowUnitQKNorm: qkNormState.allowUnitQKNorm,
    }));
  }
  if (qkNormState.wantsQKNorm) {
    await runProbes('q_norm', qTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: qTensor.dtype,
    });
    if (kTensor) {
      await runProbes('k_norm', kTensor.buffer, {
        layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
        probes: state.debugProbes, recorder,
        operatorDiagnostics: state.operatorDiagnostics, dtype: kTensor.dtype,
      });
    }
    if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
      enqueueRecordedTensorHealth(
        recorder,
        `L${layerIdx}.q_norm_HEALTH`,
        qTensor,
        qTensor.dtype,
        numTokens * numHeads * headDim
      );
      enqueueRecordedTensorHealth(
        recorder,
        `L${layerIdx}.k_norm_HEALTH`,
        kTensor,
        kTensor?.dtype ?? null,
        numTokens * numKVHeads * headDim
      );
    }
  }

  qTensor = await applyAttentionQueryScale({
    recorder, tensor: qTensor, scale: resolvedQueryScale,
    count: numTokens * numHeads * headDim,
    release: (buffer) => resourceScope.release(buffer),
    observe: (scaled) => runProbes('q_scale', scaled.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: scaled.dtype,
    }),
  });

  if (!kvCacheWriteFused && config.valueNorm === true && !reusesSharedKV) {
    const valueNormInputAliasesKey = vTensor.buffer === kTensor.buffer;
    vTensor = await applyAttentionValueNorm({
      recorder,
      vTensor,
      rmsNormEps,
      numTokens,
      numKVHeads,
      headDim,
      releaseTemporary: (buffer) => {
        if (!valueNormInputAliasesKey) {
          resourceScope.release(buffer);
        }
      },
    });
    await runProbes('v_norm', vTensor.buffer, {
      layerIdx, numTokens, hiddenSize: numKVHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, dtype: vTensor.dtype,
    });
  }

  if (normed !== attentionInput) {
    resourceScope.register(normed.buffer, 'normalized-input', 'submitOwned');
    resourceScope.release(normed.buffer);
  }
  if (attentionInputTemp) resourceScope.release(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)
  if (!ropeApplied && !disableRoPE && state.ropeFreqsCos && state.ropeFreqsSin) {
    const ropeOptions = {
      headDim,
      rotaryDim: config.ropeRotaryDim,
      pairSpanDim: config.ropeFrequencyBaseDim ?? config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      startPos: currentSeqLen,
      executionPolicies: state.executionPolicies ?? null,
    };
    if (canUseRoPEQK(qTensor, kTensor, { reusesSharedKV })) {
      await recordRoPEQK(recorder, qTensor, kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
        ...ropeOptions,
        numQHeads: numHeads,
        numKVHeads,
      });
    } else {
      await recordRoPE(recorder, qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
        ...ropeOptions,
        numHeads,
      });
      if (!reusesSharedKV) {
        await recordRoPE(recorder, kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
          ...ropeOptions,
          numHeads: numKVHeads,
        });
      }
    }
  }
  if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.q_rope_HEALTH`,
      qTensor,
      qTensor.dtype,
      numTokens * numHeads * headDim
    );
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.k_rope_HEALTH`,
      kTensor,
      kTensor?.dtype ?? null,
      numTokens * numKVHeads * headDim
    );
  }

  if (storeSharedKV && state.sharedAttentionState) {
    state.sharedAttentionState.set(layerIdx, {
      kTensor,
      vTensor,
      headDim,
      numKVHeads,
    });
  }
  resourceScope.register(qTensor.buffer, 'query', 'submitOwned');
  if (qGateTensor?.buffer) {
    resourceScope.register(qGateTensor.buffer, 'query-output-gate', 'submitOwned');
  }
  const kvOwnership = reusesSharedKV
    ? 'borrowed'
    : (storeSharedKV ? 'transferred' : 'submitOwned');
  resourceScope.register(kTensor.buffer, 'key', kvOwnership);
  resourceScope.register(vTensor.buffer, valueAliasesKey ? 'value-key-alias' : 'value', kvOwnership);

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)
  if (!skipKVCacheWrites && !diffusionGemmaDecoder && state.kvCache?.hasGPUCache?.()) {
    if (kvCacheWriteFused) {
      state.kvCache.recordF16UpdateAlreadyWrittenFromGPU(layerIdx, currentSeqLen, numTokens, tokenIds);
    } else if (state.kvCache.kvDtype === 'f16') {
      const hasExplicitF16KvContract = isAttentionKvDtypeExplicit(attentionPrecisionContract, 'f16');
      if (kTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, kTensor.dtype, 'f16', 'K would be narrowed implicitly for KV cache storage.');
      }
      if (vTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, vTensor.dtype, 'f16', 'V would be narrowed implicitly for KV cache storage.');
      }
      const canWriteDirectF32ToF16 = kTensor.dtype === 'f32'
        && vTensor.dtype === 'f32'
        && typeof state.kvCache.recordUpdateF32ToF16FromGPU === 'function';
      if (canWriteDirectF32ToF16) {
        await state.kvCache.recordUpdateF32ToF16FromGPU(
          recorder,
          layerIdx,
          kTensor.buffer,
          vTensor.buffer,
          currentSeqLen,
          numTokens,
          tokenIds
        );
      } else {
        const kCasted = kTensor.dtype === 'f16' ? kTensor : await recordCastF32ToF16(recorder, kTensor);
        const vCasted = vTensor.dtype === 'f16' ? vTensor : await recordCastF32ToF16(recorder, vTensor);

        await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens, tokenIds);

        if (kTensor.dtype !== 'f16') resourceScope.release(kCasted.buffer);
        if (vTensor.dtype !== 'f16') resourceScope.release(vCasted.buffer);
      }
    } else {
      await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens, tokenIds);
    }
  }

  // Resolve KV cache state and build dispatch parameters (shared with the immediate executor)
  let kvState;
  if (diffusionGemmaDecoder) {
    const decoderKVDtype = kvCacheDtype ?? kTensor.dtype;
    let decoderKTensor = kTensor;
    let decoderVTensor = vTensor;
    if (decoderKVDtype === 'f16' && decoderKTensor.dtype !== 'f16') {
      const hasExplicitF16KvContract = isAttentionKvDtypeExplicit(attentionPrecisionContract, 'f16');
      if (!hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, decoderKTensor.dtype, 'f16', 'DiffusionGemma decoder K would be narrowed implicitly.');
      }
      decoderKTensor = await recordCastF32ToF16(recorder, decoderKTensor);
      decoderKTemp = decoderKTensor;
    } else if (decoderKVDtype === 'f32' && decoderKTensor.dtype !== 'f32') {
      assertAttentionDtypeTransitionAllowed(state, decoderKTensor.dtype, 'f32', 'DiffusionGemma decoder K would be widened implicitly.');
      decoderKTensor = await recordCastF16ToF32(recorder, decoderKTensor);
      decoderKTemp = decoderKTensor;
    }
    if (decoderKVDtype === 'f16' && decoderVTensor.dtype !== 'f16') {
      const hasExplicitF16KvContract = isAttentionKvDtypeExplicit(attentionPrecisionContract, 'f16');
      if (!hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, decoderVTensor.dtype, 'f16', 'DiffusionGemma decoder V would be narrowed implicitly.');
      }
      decoderVTensor = await recordCastF32ToF16(recorder, decoderVTensor);
      decoderVTemp = decoderVTensor;
    } else if (decoderKVDtype === 'f32' && decoderVTensor.dtype !== 'f32') {
      assertAttentionDtypeTransitionAllowed(state, decoderVTensor.dtype, 'f32', 'DiffusionGemma decoder V would be widened implicitly.');
      decoderVTensor = await recordCastF16ToF32(recorder, decoderVTensor);
      decoderVTemp = decoderVTensor;
    }
    kvState = await createDiffusionGemmaDecoderKVState({
      state,
      layerIdx,
      kTensor: decoderKTensor,
      vTensor: decoderVTensor,
      currentSeqLen,
      numTokens,
      numKVHeads,
      headDim,
      layerType,
      slidingWindow,
      kvDtype: decoderKVDtype,
      recorder,
    });
    if (decoderKTemp) {
      resourceScope.release(decoderKTemp.buffer);
      decoderKTemp = null;
    }
    if (decoderVTemp) {
      resourceScope.release(decoderVTemp.buffer);
      decoderVTemp = null;
    }
    decoderKVState = kvState;
  } else {
    kvState = resolveKVCacheState(state, layerIdx, kTensor, vTensor, currentSeqLen, numTokens, {
      forceCurrentTensors: skipKVCacheWrites,
    });
  }
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
  attentionPlan = resolveAttentionPlanForDispatch({
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
        assertAttentionDtypeTransitionAllowed(state, qForBDPA.dtype, 'f16', 'BDPA attention would narrow Q implicitly.');
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
        assertAttentionDtypeTransitionAllowed(state, qTensor.dtype, 'f32', 'Tiered quant attention would widen Q implicitly.');
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
        assertAttentionDtypeTransitionAllowed(state, qTensor.dtype, 'f32', 'Contiguous quant attention would widen Q implicitly.');
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
          assertAttentionDtypeTransitionAllowed(state, kTensor.dtype, 'f16', 'Prefill fallback attention would narrow K implicitly.');
        }
        if (cachedVDtype === 'f16' && vTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
          assertAttentionDtypeTransitionAllowed(state, vTensor.dtype, 'f16', 'Prefill fallback attention would narrow V implicitly.');
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

  // 6. Output projection (with optional fused residual for decode)

  output = null;
  let residualFused = false;
  let oProjInput = attnForProjection;
  oProjInputTemp = null;
  if (layerWeights.oProj && getWeightBuffer) {
    ({ oProjInput, oProjInputTemp } = await prepareAttentionProjectionInput(
      attnForProjection,
      oProjInputDtype,
      (tensor) => {
        assertAttentionDtypeTransitionAllowed(
          state,
          tensor.dtype,
          oProjInputDtype,
          'Attention output projection would change activations implicitly.',
          'step_precision'
        );
        return oProjInputDtype === 'f16'
          ? recordCastF32ToF16(recorder, tensor)
          : recordCastF16ToF32(recorder, tensor);
      }
    ));
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    // Note: dtype from WeightBuffer metadata (buffer-dtypes WeakMap removed)
    const oProjDtype = getWeightDtype(oProjBuf);
    const canUseFused = attentionPlan.outputProjection.residualMode === 'matmul-residual';
    const canUseWideTileResidual =
      attentionPlan.outputProjection.residualMode === 'wide-tile-epilogue';
    if (
      attentionPlan.outputProjection.weightDtype != null
      && oProjDtype !== attentionPlan.outputProjection.weightDtype
    ) {
      throw new Error(
        `Attention output projection dtype changed after planning at layer ${layerIdx}: ` +
        `plan=${attentionPlan.outputProjection.weightDtype}, bound=${oProjDtype}.`
      );
    }
    if (canUseFused && residualTensor) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await recordMatmulResidualFused(recorder, oProjInput, oProjBuf, residualTensor, {
        N: hiddenSize,
        K: numHeads * headDim,
      });
      residualFused = true;
    } else {
      // STANDARD PATH: o_proj matmul only unless Q4K WideTile fuses residual in the epilogue
      output = await recordMatmul(recorder, oProjInput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
        transposeB: 'auto',
        role: 'o_proj',
        layerIdx,
        kernelPath,
        outputDtype: oProjOutputDtype,
        executionPolicies: state.executionPolicies ?? null,
        residualTensor: canUseWideTileResidual ? residualTensor : null,
      });
      residualFused = canUseWideTileResidual;
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!isGpuBufferInstance(layerWeights.oProj) && !isWeightBuffer(layerWeights.oProj)) {
      resourceScope.release(isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
    }
  } else {
    output = oProjInput;
  }

  // Apply LoRA to output projection if present (only if not using fused path)
  if (!residualFused) {
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');
    if (loraO && getWeightBuffer) {
      const combined = await applyLoRA(
        oProjInput,
        output,
        loraO,
        { M: numTokens, N: hiddenSize, K: numHeads * headDim },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== output.buffer) {
        resourceScope.release(output.buffer);
        output = combined;
      }
    }
  }

  if (layerWeights.oProjBias) {
    const { tensor: oProjBiasTensor, owned } = getVectorTensor(
      layerWeights.oProjBias,
      `L${layerIdx}.o_proj_bias`,
      hiddenSize,
      {},
      debugFlags
    );
    output = await recordBiasAdd(
      recorder,
      output,
      oProjBiasTensor,
      numTokens,
      hiddenSize,
      {
        label: `L${layerIdx}.o_proj_bias`,
        layerIdx,
        executionPolicies: state.executionPolicies ?? null,
      }
    );
    if (owned) resourceScope.release(oProjBiasTensor.buffer);
  }

  finalOutput = output;
  if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
    enqueueRecordedTensorHealth(
      recorder,
      `L${layerIdx}.o_proj_HEALTH`,
      output,
      output.dtype,
      numTokens * hiddenSize
    );
  }

  const buffersToTrack = [];
  if (output.buffer !== attnForProjection.buffer) {
    buffersToTrack.push(attnForProjection.buffer);
  }
  if (oProjInputTemp && oProjInputTemp.buffer !== attnForProjection.buffer) {
    buffersToTrack.push(oProjInputTemp.buffer);
  }
  if (output.dtype !== oProjOutputDtype) {
    assertAttentionDtypeTransitionAllowed(state, output.dtype, oProjOutputDtype, 'Attention output would change implicitly before leaving the layer.');
    const coercedOutput = oProjOutputDtype === 'f16'
      ? await recordCastF32ToF16(recorder, output)
      : await recordCastF16ToF32(recorder, output);
    buffersToTrack.push(output.buffer);
    finalOutput = coercedOutput;
    resourceScope.register(finalOutput.buffer, 'coerced-attention-output', 'submitOwned');
  }

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
  return { output: finalOutput, residualFused };
  } catch (error) {
    const trackOnce = (buffer) => resourceScope.release(buffer, 'attention-failure');
    if (finalOutput?.buffer && finalOutput.buffer !== output?.buffer) {
      trackOnce(finalOutput.buffer);
    }
    if (output?.buffer && output.buffer !== attnForProjection?.buffer) {
      trackOnce(output.buffer);
    }
    if (oProjInputTemp?.buffer) {
      trackOnce(oProjInputTemp.buffer);
    }
    if (attnForProjection?.buffer && attnForProjection.buffer !== attnOutput?.buffer) {
      trackOnce(attnForProjection.buffer);
    }
    if (attnOutput?.buffer) {
      trackOnce(attnOutput.buffer);
    }
    if (qGateTensor?.buffer) {
      trackOnce(qGateTensor.buffer);
    }
    if (qTensor?.buffer) {
      trackOnce(qTensor.buffer);
    }
    if (kTensor?.buffer && !retainSharedKvBuffers) {
      trackOnce(kTensor.buffer);
    }
    if (vTensor?.buffer && !retainSharedKvBuffers) {
      trackOnce(vTensor.buffer);
    }
    if (normed?.buffer && normed.buffer !== attentionInput?.buffer) {
      trackOnce(normed.buffer);
    }
    if (attentionInputTemp && attentionInput?.buffer) {
      trackOnce(attentionInput.buffer);
    }
    if (decoderKTemp?.buffer) {
      trackOnce(decoderKTemp.buffer);
    }
    if (decoderVTemp?.buffer) {
      trackOnce(decoderVTemp.buffer);
    }
    if (decoderKVState?.ownedBuffers) {
      for (const buffer of decoderKVState.ownedBuffers) {
        trackOnce(buffer);
      }
      decoderKVState.ownedBuffers = null;
    }
    const resourceEvents = resourceScope.close('failure');
    if (state.observationContext?.receiptPolicy !== 'off') {
      state.stats.attentionResourceEvents ??= [];
      state.stats.attentionResourceEvents.push(...resourceEvents);
    }
    captureAttentionRefactorReceipt({
      state,
      plan: attentionPlan,
      resourceEvents,
      error,
      failureBoundary: attentionPlan ? 'execute-bound-attention-plan' : 'bind-attention-plan',
    });
    throw error;
  }
}
