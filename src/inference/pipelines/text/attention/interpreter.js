import { recordAttentionOutputProjection } from "./interpreter/output.js";
import { assertPairedAttentionExecution } from "./paired-contract.js";
import { recordAttentionCacheCast } from "./cache-cast.js";
import { isGpuBufferInstance, isWeightBuffer, requireWeightDtype } from "../../../../gpu/weight-buffer.js";

import { acquireBuffer } from "../../../../memory/buffer-pool.js";

import { recordRMSNorm } from "../../../../gpu/kernels/rmsnorm.js";
import { recordLayerNorm } from "../../../../gpu/kernels/layernorm.js";

import { recordRoPE } from "../../../../gpu/kernels/rope.js";
import { canUseRoPEQK, recordRoPEQK } from "../../../../gpu/kernels/rope-qk.js";

import { recordCastF16ToF32, recordCastF32ToF16 } from "../../../../gpu/kernels/cast.js";

import { createTensor } from "../../../../gpu/tensor.js";

import { trace } from "../../../../debug/index.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";
import { shouldForceF32AttentionProjectionForRoPE, resolveAttentionProjectionOutputDtype, projectAttentionQKV, applyAttentionQKNorm, applyAttentionValueNorm, hasAttentionProjectionDiagnostics, hasAttentionStageDiagnostics, resolveAttentionQKNormState } from "./projections.js";

import { runProbes, tensorProbeOptions } from "../observation/probes.js";
import { shouldDebugLayer } from "./types.js";
import { getKernelPathMatmulPrecision, getKernelPathMatmulVariant } from "../../../../config/kernel-path-loader.js";
import { resolveKVCacheState, createDiffusionGemmaDecoderKVState } from "./dispatch-params.js";

import { resolveAttentionPrecisionContract, isAttentionKvDtypeExplicit } from "./precision-contract.js";
import { canUseRmsNormWideTileProjectionFusion } from "./rmsnorm-fusion-gate.js";
import { getVectorTensor } from "../loading/weights.js";
import { resolveAttentionRuntimeSession } from "../configuration/resolved-runtime-session.js";

import { createRecordedResourceScope } from "../../../resource-scope.js";

import { observeAttentionRoPE } from "./rope-observation.js";
import { applyAttentionQueryScale } from "./query-transform.js";
import { resolveQueryScale } from "../../../../config/heterogeneous-attention-contract.js";
import { observeRecordedAttentionProjections, observeRecordedAttentionQKNorm, shouldTraceRecordedHealth } from "./recorded-health.js";
import { failRecordedAttention, resolveDirectF16KVCacheWrite, runRecordedAttentionCore } from "./interpreter/recorded.js";
import { assertAttentionDtypeTransitionAllowed, resolveRecordedAttentionDispatch } from "./interpreter/immediate.js";
const ATTENTION_DTYPE_LOGGED = new Set();
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
  assertPairedAttentionExecution(input, config, state, runtimeSession, layerWeights);
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
    ? 'step-precision'
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
  const canFuseInputNormProjRec = input.precision !== 'two-f32' && rmsNormFusionFlagRec
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
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(normed),
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
        { batchSize: numTokens, hiddenSize, label: 'input_norm', normWeightDtype: requireWeightDtype(normWeightBuf, `L${layerIdx}.input_norm`) }
      );
    } else {
      normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
        label: 'input_norm', kernelPath, phase, layerIdx, role: 'input_norm', section: 'layer',
      });
    }
    if (normBiasOwned && normBiasTensor) resourceScope.release(normBiasTensor.buffer);
    if (!isGpuBufferInstance(layerWeights.inputNorm) && !isWeightBuffer(layerWeights.inputNorm)) resourceScope.release(normWeightBuf);
    await runProbes('post_input_norm', normed.buffer, {
      layerIdx, numTokens, hiddenSize,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(normed),
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
  const directF16KVCacheWrite = skipKVCacheWrites || input.precision === 'two-f32'
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
  await observeRecordedAttentionProjections({ qTensor, kTensor, vTensor, config, state, debugFlags, recorder });
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
      headDim, kernelPath, phase, layerIdx,
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
    await observeRecordedAttentionQKNorm({ qTensor, kTensor, config, state, debugFlags, recorder });
  }
  qTensor = await applyAttentionQueryScale({
    recorder, tensor: qTensor, scale: resolvedQueryScale,
    count: numTokens * numHeads * headDim,
    release: (buffer) => resourceScope.release(buffer),
    observe: (scaled) => runProbes('q_scale', scaled.buffer, {
      layerIdx, numTokens, hiddenSize: numHeads * headDim,
      probes: state.debugProbes, recorder,
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(scaled),
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
      operatorDiagnostics: state.operatorDiagnostics, ...tensorProbeOptions(vTensor),
    });
  }
  if (normed !== attentionInput) {
    resourceScope.register(normed.buffer, 'normalized-input', 'submitOwned');
    resourceScope.release(normed.buffer);
  }
  if (attentionInputTemp) resourceScope.release(attentionInput.buffer);
  if (!ropeApplied && !disableRoPE && state.ropeFreqsCos && state.ropeFreqsSin) {
    const ropeOptions = {
      frequencyPrecision: runtimeSession.intermediatePrecision,
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
  await observeAttentionRoPE({ state, recorder, ropeApplied, disableRoPE, qTensor, kTensor, layerIdx, numTokens, numHeads, numKVHeads, headDim });
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
  // Direct cache writes produce no temporary K/V tensors to own.
  if (kTensor) resourceScope.register(kTensor.buffer, 'key', kvOwnership);
  if (vTensor) resourceScope.register(vTensor.buffer, valueAliasesKey ? 'value-key-alias' : 'value', kvOwnership);

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
      const canWriteDirectF32ToF16 = kTensor.precision !== 'two-f32' && vTensor.precision !== 'two-f32' && kTensor.dtype === 'f32'
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
        const kCasted = await recordAttentionCacheCast(recorder, kTensor, resourceScope, 'cache-key-cast');
        const vCasted = await recordAttentionCacheCast(recorder, vTensor, resourceScope, 'cache-value-cast');

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
  const attentionDispatch = resolveRecordedAttentionDispatch({
    attentionActivationDtype, attentionPrecisionContract, canFuseInputNormProjRec,
    config, diffusionGemmaDecoder, headDim, hiddenSize, input, isPrefill,
    kTensor, kernelPath, kvCacheDtype, kvState, layerIdx, layerType, layerWeights, lora,
    matmulOutputDtype, normed, numHeads, numKVHeads, numTokens,
    oProjInputDtype, oProjOutputDtype, qGateTensor, qTensor, qkNormApplied,
    queryPreAttnScalar, reusesSharedKV, ropeApplied, runtimeSession,
    skipKVCacheWrites, slidingWindow, state, storeSharedKV, usedFusedQKV,
    useF16Activations, vTensor,
  });
  attentionPlan = attentionDispatch.attentionPlan;
  const {
    attentionKernelVariant,
    attentionKernelPath,
    attnScale,
    cachedKTensor,
    cachedKDtype,
    cachedVTensor,
    cachedVDtype,
    causalForAttention,
    effectiveSlidingWindow,
    prefillFallbackNeedsCast,
  } = attentionDispatch;
  ({ attnOutput, attnForProjection } = await runRecordedAttentionCore({
    attentionKernelVariant,
    attentionPlan,
    attentionPrecisionContract,
    attentionKernelPath,
    assertDtypeTransition: (fromDtype, toDtype, detail, transitionDeclaredBy = null) =>
      assertAttentionDtypeTransitionAllowed(state, fromDtype, toDtype, detail, transitionDeclaredBy),
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
  }));

  return await recordAttentionOutputProjection({
    attnForProjection,
    layerWeights,
    getWeightBuffer,
    oProjInputDtype,
    state,
    recorder,
    lora,
    layerIdx,
    attentionPlan,
    residualTensor,
    hiddenSize,
    numHeads,
    headDim,
    numTokens,
    kernelPath,
    oProjOutputDtype,
    resourceScope,
    debugFlags,
    kTensor,
    qGateTensor,
    qTensor,
    retainSharedKvBuffers,
    vTensor
  });
  } catch (error) {
    return failRecordedAttention({
      resourceScope,
      attnForProjection,
      attnOutput,
      qGateTensor,
      qTensor,
      kTensor,
      retainSharedKvBuffers,
      vTensor,
      normed,
      attentionInput,
      attentionInputTemp,
      decoderKTemp,
      decoderVTemp,
      decoderKVState,
      state,
      attentionPlan,
      error
    });
  }
}
