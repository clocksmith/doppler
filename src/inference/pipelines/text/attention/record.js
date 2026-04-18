

import { isGpuBufferInstance, isWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import {
  recordMatmul,
  recordRMSNorm,
  recordRoPE,
  recordAttention,
  recordAttentionTiered,
  recordAttentionTieredQuant,
  recordAttentionContiguousQuant,
  recordAttentionBDPA,
  recordSiLU,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
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
} from './projections.js';
import { prepareAttentionProjectionInput } from './output-projection.js';

import { releaseOrTrack, shouldDebugLayer } from './types.js';
import {
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../config/kernel-path-loader.js';
import {
  resolveKVCacheState,
  buildAttentionDispatchParams,
  buildAttentionInputsData,
} from './dispatch-params.js';
import {
  buildTieredQuantAttentionOptions,
  buildContiguousQuantAttentionOptions,
} from './quant-options.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import {
  resolveAttentionPrecisionContract,
  isAttentionKvDtypeExplicit,
} from './precision-contract.js';

const ATTENTION_DTYPE_LOGGED = new Set();

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


export async function recordLayerAttentionGPU(
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
    skipInputNorm = false,
    tokenIds = null,
    kernelPath = null,
    disableRoPE = false,
    multimodalBidirectionalSpan = null,
    sharedKVSourceLayerIdx = null,
    storeSharedKV = false,
  } = config;

  const phase = isPrefill ? 'prefill' : 'decode';
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
  if (!allowF16Attention && input.dtype !== attentionActivationDtype) {
    assertAttentionDtypeTransitionAllowed(
      state,
      input.dtype,
      attentionActivationDtype,
      'The attention kernel selection would widen the input implicitly.'
    );
    attentionInput = attentionActivationDtype === 'f16'
      ? await recordCastF32ToF16(recorder, input)
      : await recordCastF16ToF32(recorder, input);
    attentionInputTemp = true;
    normed = attentionInput;
  }

  if (!layerWeights) {
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const output = createTensor(outputBuf, oProjOutputDtype, [numTokens, hiddenSize], 'attn_output');
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
  // state.runtimeConfig is a stale snapshot whose inference.session is an empty
  // object; the live module-level runtime config carries the merged
  // profile/override session. Read session-level flags from getRuntimeConfig().
  const rmsNormFusionFlagRec = getRuntimeConfig()?.inference?.session?.useFusedRmsnormWideTile === true;
  const canFuseInputNormProjRec = rmsNormFusionFlagRec
    && !skipInputNorm
    && layerWeights.inputNorm
    && getNormWeightBuffer
    && isPrefill
    && numTokens > 1
    && attentionInput.dtype === 'f32';

  try {
  if (canFuseInputNormProjRec) {
    fusedNormWeightRec = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    fusedNormEpsRec = rmsNormEps;
    fusedNormOffsetRec = config.rmsNormWeightOffset === true;
    fusedNormOwnedRec = !isGpuBufferInstance(layerWeights.inputNorm) && !isWeightBuffer(layerWeights.inputNorm);
    // Keep normed = attentionInput (raw). Each q/k/v_proj matmul runs norm internally.
  } else if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    if (!isGpuBufferInstance(layerWeights.inputNorm) && !isWeightBuffer(layerWeights.inputNorm)) releaseOrTrack(recorder, normWeightBuf);
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
  ({ qTensor, qGateTensor, kTensor, vTensor, usedFusedQKV } = await projectAttentionQKV({
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
    releaseTemporary: (buffer) => releaseOrTrack(recorder, buffer),
    onFusedQKV: layerIdx === 0 && isPrefill
      ? ({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize }) => {
        trace.attn(layerIdx, `Using fused QKV path: ${qSizeFused}+${kSizeFused}+${vSizeFused}=${totalSize}`);
      }
      : null,
    fusedNormWeight: fusedNormWeightRec,
    fusedNormEps: fusedNormEpsRec,
    fusedNormOffset: fusedNormOffsetRec,
  }));
  // Deferred release of the norm weight buffer for fused path.
  if (fusedNormWeightRec && fusedNormOwnedRec) {
    releaseOrTrack(recorder, fusedNormWeightRec);
  }

  // Optional per-head Q/K normalization.
  // Some models use RMSNorm with (1+weight) offset formula, controlled by rmsNormWeightOffset.
  const wantsQKNorm = config.queryKeyNorm === true;
  if (wantsQKNorm && layerIdx === 0 && (!layerWeights.qNorm || !layerWeights.kNorm)) {
    log.warn('Attention', `Q/K norm requested but weights missing (hasQ=${!!layerWeights.qNorm}, hasK=${!!layerWeights.kNorm}); skipping QK norm.`);
  }
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
    rmsNormWeightOffset: config.rmsNormWeightOffset,
    releaseTemporary: (buffer) => releaseOrTrack(recorder, buffer),
    skipKNorm: reusesSharedKV,
  }));

  if (config.valueNorm === true && !reusesSharedKV) {
    vTensor = await applyAttentionValueNorm({
      recorder,
      vTensor,
      rmsNormEps,
      numTokens,
      numKVHeads,
      headDim,
      releaseTemporary: (buffer) => releaseOrTrack(recorder, buffer),
    });
  }

  if (normed !== attentionInput) releaseOrTrack(recorder, normed.buffer);
  if (attentionInputTemp) recorder.trackTemporaryBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)
  if (!disableRoPE && state.ropeFreqsCos && state.ropeFreqsSin) {
    await recordRoPE(recorder, qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads,
      headDim,
      rotaryDim: config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      startPos: currentSeqLen,
      executionPolicies: state.executionPolicies ?? null,
    });
    if (!reusesSharedKV) {
      await recordRoPE(recorder, kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
        numHeads: numKVHeads,
        headDim,
        rotaryDim: config.ropeRotaryDim,
        interleaved: config.ropeInterleaved,
        startPos: currentSeqLen,
        executionPolicies: state.executionPolicies ?? null,
      });
    }
  }

  if (storeSharedKV && state.sharedAttentionState) {
    state.sharedAttentionState.set(layerIdx, {
      kTensor,
      vTensor,
      headDim,
      numKVHeads,
    });
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)
  if (state.kvCache?.hasGPUCache?.()) {
    // Use recordUpdateFromGPU to record copy operations to the recorder's encoder
    // This ensures K/V buffers are populated before copying (all ops submitted together)
    if (state.kvCache.kvDtype === 'f16') {
      const hasExplicitF16KvContract = isAttentionKvDtypeExplicit(attentionPrecisionContract, 'f16');
      if (kTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, kTensor.dtype, 'f16', 'K would be narrowed implicitly for KV cache storage.');
      }
      if (vTensor.dtype !== 'f16' && !hasExplicitF16KvContract) {
        assertAttentionDtypeTransitionAllowed(state, vTensor.dtype, 'f16', 'V would be narrowed implicitly for KV cache storage.');
      }
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await recordCastF32ToF16(recorder, kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await recordCastF32ToF16(recorder, vTensor);

      await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens, tokenIds);

      if (kTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(vCasted.buffer);
    } else {
      await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens, tokenIds);
    }
  }

  // Resolve KV cache state and build dispatch parameters (shared with run.js)
  const kvState = resolveKVCacheState(state, layerIdx, kTensor, vTensor, currentSeqLen, numTokens);
  const dispatchConfig = {
    layerIdx, numTokens, isPrefill, numHeads, numKVHeads, headDim, hiddenSize,
    slidingWindow, layerType, layerTypes: config.layerTypes,
    queryPreAttnScalar, causalAttention: config.causalAttention,
    activationDtype: attentionActivationDtype,
    kvCacheDtype: attentionPrecisionContract.resolvedKvCacheDtype ?? state.kvCache?.kvDtype ?? null,
  };
  const dispatchParams = buildAttentionDispatchParams(dispatchConfig, state, kTensor, vTensor, kvState);
  const {
    effectiveSlidingWindow, attentionKernelVariant, attnScale,
    cachedKDtype, cachedVDtype, cachedKTensor, cachedVTensor,
    prefillFallbackNeedsCast, causalForAttention,
  } = dispatchParams;

  // 5. Attention

  recordAttentionInputs(state, buildAttentionInputsData(
    dispatchConfig, input, normed, kvState, dispatchParams,
    { useF16Activations, matmulOutputDtype },
    usedFusedQKV, qTensor, kTensor, vTensor,
  ));

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
        recorder.trackTemporaryBuffer(qForBDPA.buffer);
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
        recorder.trackTemporaryBuffer(qForAttention.buffer);
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
        recorder.trackTemporaryBuffer(qForAttention.buffer);
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
        if (kTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(kCasted.buffer);
        if (vTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(vCasted.buffer);
      }
      // Session precedence is runtime-over-manifest per config-style-guide
      // §Category Rules. getRuntimeConfig() returns the merged session (manifest
      // is the base layer, runtime fields win field-by-field via merge.js).
      // Kernel enforces head_dim=256, f16 KV, contiguous layout; only applies
      // when numTokens > 1 (prefill). Same flag semantics as the non-recorder
      // path in ./run.js.
      const mergedSessionRec = getRuntimeConfig()?.inference?.session;
      const useFlashPrefillRec = mergedSessionRec?.useFlashPrefillAttention === true && numTokens > 1;
      const useOrtFlashPrefillRec = mergedSessionRec?.useOrtFlashPrefillAttention === true && numTokens > 1;
      return recordAttention(recorder, qTensor, kForAttn, vForAttn, null, numHeads, headDim, {
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
        kernelPath,
        useFlashPrefill: useFlashPrefillRec,
        useOrtFlashPrefill: useOrtFlashPrefillRec,
      });
    },
  };
  const runAttentionKernel = attentionKernelRunners[attentionKernelVariant];
  if (!runAttentionKernel) {
    throw new Error(`Unsupported attention kernel variant "${attentionKernelVariant}" at layer ${layerIdx}`);
  }

  attnOutput = await runAttentionKernel();

  attnForProjection = attnOutput;
  if (qGateTensor) {
    attnForProjection = await recordSiLU(recorder, attnOutput, {
      size: numTokens * numHeads * headDim,
      gate: qGateTensor,
      gateActivation: 'sigmoid',
      inputActivation: 'identity',
      swigluLimit: null,
    });
    recorder.trackTemporaryBuffer(attnOutput.buffer);
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
    const canUseFused = selectRuleValue('inference', 'attention', 'useFusedOProjResidual', {
      allowFusedResidual: shouldUseFusedMatmulResidual(numTokens),
      hasResidual: Boolean(residualTensor),
      residualMatches: Boolean(residualTensor && residualTensor.dtype === oProjInput.dtype),
      attnIsF32: oProjInput.dtype === 'f32',
      attnIsF16: oProjInput.dtype === 'f16',
      hasLoRA: Boolean(loraO),
      oProjIsF16: oProjDtype === 'f16',
    });

    if (canUseFused && residualTensor) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await recordMatmulResidualFused(recorder, oProjInput, oProjBuf, residualTensor, {
        N: hiddenSize,
        K: numHeads * headDim,
      });
      residualFused = true;
    } else {
      // STANDARD PATH: o_proj matmul only
      output = await recordMatmul(recorder, oProjInput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
        transposeB: 'auto',
        role: 'o_proj',
        layerIdx,
        kernelPath,
        outputDtype: oProjOutputDtype,
        executionPolicies: state.executionPolicies ?? null,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!isGpuBufferInstance(layerWeights.oProj) && !isWeightBuffer(layerWeights.oProj)) {
      releaseOrTrack(recorder, isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
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
        recorder.trackTemporaryBuffer(output.buffer);
        output = combined;
      }
    }
  }

  finalOutput = output;

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
  }

  // Track intermediate buffers for cleanup after submit (not release!)
  // These buffers are used by recorded operations that haven't executed yet.
  // Releasing them back to the pool would allow reuse before the encoder is submitted,
  // causing data corruption (especially for small decode buffers).
  recorder.trackTemporaryBuffer(qTensor.buffer);
  if (qGateTensor) {
    recorder.trackTemporaryBuffer(qGateTensor.buffer);
  }
  if (!retainSharedKvBuffers) {
    recorder.trackTemporaryBuffer(kTensor.buffer);
    recorder.trackTemporaryBuffer(vTensor.buffer);
  }
  for (const buffer of buffersToTrack) {
    recorder.trackTemporaryBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
  } catch (error) {
    const tracked = new Set();
    const trackOnce = (buffer) => {
      if (!buffer || tracked.has(buffer)) return;
      tracked.add(buffer);
      recorder.trackTemporaryBuffer(buffer);
    };
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
    throw error;
  }
}
