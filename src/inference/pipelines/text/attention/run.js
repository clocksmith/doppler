

import { isWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runRoPE,
  runAttention,
  runAttentionBDPA,
  runAttentionTieredQuant,
  runAttentionTiered,
  runAttentionContiguousQuant,
  runSiLU,
  castF16ToF32,
  castF32ToF16,
  runMatmulResidualFused,
  shouldUseFusedMatmulResidual,
} from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { isKernelDebugEnabled, dumpTokenVector, dumpKVCache, logKernelStep } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace } from '../../../../debug/index.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { runProbes } from '../probes.js';
import {
  recordAttentionInputs,
  shouldForceF32AttentionProjectionForRoPE,
  resolveAttentionProjectionOutputDtype,
  projectAttentionQKV,
  applyAttentionQKNorm,
  applyAttentionValueNorm,
} from './projections.js';
import { prepareAttentionProjectionInput } from './output-projection.js';

import {
  shouldDebugLayer,
  markStageLogged,
} from './types.js';
import { getKernelPathMatmulVariant } from '../../../../config/kernel-path-loader.js';
import {
  resolveKVCacheState,
  buildAttentionDispatchParams,
  buildAttentionInputsData,
} from './dispatch-params.js';
import {
  buildTieredQuantAttentionOptions,
  buildContiguousQuantAttentionOptions,
} from './quant-options.js';

const ATTENTION_DTYPE_LOGGED = new Set();

export async function runLayerAttentionGPU(
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
  } = config;

  const device = getDevice();

  const desiredOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: config.activationDtype,
  });
  const wantsF16Output = desiredOutputDtype === 'f16';
  const useF16Activations = wantsF16Output;
  const kvCacheFallback = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
  const kvCacheDtype = state.kvCache?.kvDtype ?? kvCacheFallback;
  const allowF16Attention = wantsF16Output && kvCacheDtype === 'f16';
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
  if (wantsF16Output && !allowF16Attention) {
    attentionInput = await castF16ToF32(input);
    attentionInputTemp = true;
    normed = attentionInput;
  }

  // Debug: attention input for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_input', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(attentionInput.buffer, `L${layerIdx} attention input (GPU)`, numTokens, hiddenSize);
  }

  // Debug logging moved to debug-utils.ts (enable via setDebugConfig)

  if (!layerWeights) {
    // Return zeros if no weights
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const outputDtype = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
    const output = createTensor(outputBuf, outputDtype, [numTokens, hiddenSize], 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  
  try {
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    try {
      // Debug: norm weights for configured layers
      if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'norm_weights', debugFlags) && debugCheckBuffer) {
        await debugCheckBuffer(normWeightBuf, `L${layerIdx} input norm weights (GPU)`, 1, hiddenSize);
      }

      normed = await runRMSNorm(attentionInput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
    } finally {
      if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    }

    // Trace input norm output
    if (kernelTrace.enabled) {
      await traceStep('rmsnorm', `L${layerIdx}.input_norm`, layerIdx, normed.buffer, [numTokens, hiddenSize]);
    }

    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('rmsnorm', { layerIdx, label: 'input_norm', size: numTokens * hiddenSize });
      await dumpTokenVector(normed.buffer, 'input_norm_out', {
        layerIdx,
        tokenIdx: Math.max(0, numTokens - 1),
        rowSize: hiddenSize,
        dtype: normed.dtype,
      });
    }

    await runProbes('post_input_norm', normed.buffer, {
      layerIdx,
      numTokens,
      hiddenSize,
      probes: state.debugProbes,
      operatorDiagnostics: state.operatorDiagnostics,
      dtype: normed.dtype,
    });
  }

  // Debug: Check normed input for L0 prefill
  // Debug: normed input for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_normed', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(normed.buffer, `L${layerIdx} normed input (GPU)`, numTokens);
  }

  const debugLayers = debugFlags.debugLayers;
  const shouldLogLayer = debugLayers === null ? layerIdx === 0 : shouldDebugLayer(layerIdx, debugLayers);
  if (shouldLogLayer) {
    const phase = selectRuleValue('kernels', 'attention', 'phase', { isDecode: !isPrefill });
    const logKey = `L${layerIdx}_${phase}_dtypes`;
    if (!ATTENTION_DTYPE_LOGGED.has(logKey)) {
      ATTENTION_DTYPE_LOGGED.add(logKey);
      trace.attn(layerIdx, `dtypes: activation=${config.activationDtype ?? 'unknown'}, input=${input.dtype}, normed=${normed.dtype}`);
    }
  }

  if (isKernelDebugEnabled(layerIdx)) {
    await dumpTokenVector(normed.buffer, 'attn_in', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: hiddenSize,
      dtype: normed.dtype,
    });
  }

  // 2. Q/K/V projections
  const qProjVariant = getKernelPathMatmulVariant('q_proj', isPrefill ? 'prefill' : 'decode', layerIdx, kernelPath);
  const kernelPathIsF16 = qProjVariant != null && qProjVariant.includes('f16') && !qProjVariant.includes('f32');
  const matmulOutputDtype = resolveAttentionProjectionOutputDtype(desiredOutputDtype, {
    forceF32: shouldForceF32AttentionProjectionForRoPE({
      attentionInputDtype: desiredOutputDtype,
      headDim,
      rotaryDim: config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      kernelPathIsF16,
    }),
  });
  let usedFusedQKV = false;
  ({ qTensor, qGateTensor, kTensor, vTensor, usedFusedQKV } = await projectAttentionQKV({
    recorder: null,
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
    releaseTemporary: (buffer) => releaseBuffer(buffer),
    onFusedQKV: layerIdx === 0 && isPrefill
      ? ({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize }) => {
        trace.attn(layerIdx, `Using fused QKV path: ${qSizeFused}+${kSizeFused}+${vSizeFused}=${totalSize}`);
      }
      : null,
  }));

  // Trace Q/K/V projections
  if (kernelTrace.enabled) {
    await traceStep('matmul', `L${layerIdx}.q_proj`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.k_proj`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.v_proj`, layerIdx, vTensor.buffer, [numTokens, numKVHeads * headDim]);
  }
  await runProbes('q_proj', qTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: qTensor.dtype,
  });
  await runProbes('k_proj', kTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numKVHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: kTensor.dtype,
  });
  await runProbes('v_proj', vTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numKVHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: vTensor.dtype,
  });

  // Kernel step debug: Q/K/V projections
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('matmul', { layerIdx, label: 'Q_proj', M: numTokens, N: numHeads * headDim, K: hiddenSize });
    await dumpTokenVector(qTensor.buffer, 'Q_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: qTensor.dtype,
    });
    logKernelStep('matmul', { layerIdx, label: 'K_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(kTensor.buffer, 'K_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: kTensor.dtype,
    });
    logKernelStep('matmul', { layerIdx, label: 'V_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(vTensor.buffer, 'V_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: vTensor.dtype,
    });
  }

  // Debug: Check Q/K/V after projections for L0 prefill
  // Debug: Q/K/V projections for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'qkv_proj', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(qTensor.buffer, `L${layerIdx} Q after proj (GPU)`, numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, `L${layerIdx} K after proj (GPU)`, numTokens, numKVHeads * headDim);
    await debugCheckBuffer(vTensor.buffer, `L${layerIdx} V after proj (GPU)`, numTokens, numKVHeads * headDim);
  }

  // Optional per-head Q/K normalization
  const wantsQKNorm = config.queryKeyNorm === true;
  const hasQNorm = !!layerWeights.qNorm;
  const hasKNorm = !!layerWeights.kNorm;
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('qk_norm', { layerIdx, label: `hasQ=${hasQNorm} hasK=${hasKNorm} wants=${wantsQKNorm}` });
  }
  if (wantsQKNorm && layerIdx === 0 && (!hasQNorm || !hasKNorm)) {
    log.warn('Attention', `Q/K norm requested but weights missing (hasQ=${hasQNorm}, hasK=${hasKNorm}); skipping QK norm.`);
  }

  ({ qTensor, kTensor } = await applyAttentionQKNorm({
    recorder: null,
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
    releaseTemporary: (buffer) => releaseBuffer(buffer),
    onQNormApplied: isKernelDebugEnabled(layerIdx)
      ? async (tensor) => {
        await dumpTokenVector(tensor.buffer, 'Q_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numHeads * headDim,
          dtype: tensor.dtype,
        });
      }
      : null,
    onKNormApplied: isKernelDebugEnabled(layerIdx)
      ? async (tensor) => {
        await dumpTokenVector(tensor.buffer, 'K_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numKVHeads * headDim,
          dtype: tensor.dtype,
        });
      }
    : null,
  }));

  if (config.valueNorm === true) {
    vTensor = await applyAttentionValueNorm({
      recorder: null,
      vTensor,
      rmsNormEps,
      numTokens,
      numKVHeads,
      headDim,
      releaseTemporary: (buffer) => releaseBuffer(buffer),
    });
  }

  await runProbes('q_norm', qTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: qTensor.dtype,
  });
  await runProbes('k_norm', kTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numKVHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: kTensor.dtype,
  });

  if (normed !== attentionInput) releaseBuffer(normed.buffer);
  if (attentionInputTemp) releaseBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)

  if (!disableRoPE && state.ropeFreqsCos && state.ropeFreqsSin) {
    await runRoPE(qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads,
      headDim,
      rotaryDim: config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      startPos: currentSeqLen,
    });
    await runRoPE(kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads,
      headDim,
      rotaryDim: config.ropeRotaryDim,
      interleaved: config.ropeInterleaved,
      startPos: currentSeqLen,
    });

    // Trace RoPE outputs
    if (kernelTrace.enabled) {
      await traceStep('rope', `L${layerIdx}.q_rope`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
      await traceStep('rope', `L${layerIdx}.k_rope`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    }
  }
  await runProbes('q_rope', qTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: qTensor.dtype,
  });
  await runProbes('k_rope', kTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: numKVHeads * headDim,
    probes: state.debugProbes,
    operatorDiagnostics: state.operatorDiagnostics,
    dtype: kTensor.dtype,
  });
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('rope', { layerIdx, label: `startPos=${currentSeqLen}` });
    await dumpTokenVector(qTensor.buffer, 'Q_rope', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: qTensor.dtype,
    });
    await dumpTokenVector(kTensor.buffer, 'K_rope', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: kTensor.dtype,
    });
  }

  // Debug: Check Q/K after RoPE for L0 prefill
  // Debug: Q/K after RoPE for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'qk_rope', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(qTensor.buffer, `L${layerIdx} Q after RoPE (GPU)`, numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, `L${layerIdx} K after RoPE (GPU)`, numTokens, numKVHeads * headDim);
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)

  if (state.kvCache?.hasGPUCache?.()) {
    if (state.kvCache.kvDtype === 'f16') {
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await castF32ToF16(kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await castF32ToF16(vTensor);

      await state.kvCache.updateFromGPU(layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens, tokenIds);

      if (kTensor.dtype !== 'f16') releaseBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') releaseBuffer(vCasted.buffer);
    } else {
      await state.kvCache.updateFromGPU(layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens, tokenIds);
    }

    // Kernel step debug: KV cache state after update
    if (isKernelDebugEnabled(layerIdx)) {
      trace.kv(layerIdx, `KV cache updated: startPos=${currentSeqLen}, numTokens=${numTokens}`);
      await dumpKVCache(( (state.kvCache)), layerIdx);
    }
  }

  // Resolve KV cache state and build dispatch parameters (shared with record.js)
  const kvState = resolveKVCacheState(state, layerIdx, kTensor, vTensor, currentSeqLen, numTokens);
  const dispatchConfig = {
    layerIdx, numTokens, isPrefill, numHeads, numKVHeads, headDim, hiddenSize,
    slidingWindow, layerType, layerTypes: config.layerTypes,
    queryPreAttnScalar, causalAttention: config.causalAttention,
    activationDtype: config.activationDtype,
    kvCacheDtype: state.kvCache?.kvDtype ?? null,
  };
  const dispatchParams = buildAttentionDispatchParams(dispatchConfig, state, kTensor, vTensor, kvState);
  const {
    effectiveSlidingWindow, attentionKernelVariant, attnScale,
    cachedKDtype, cachedVDtype, cachedKTensor, cachedVTensor,
    prefillFallbackNeedsCast, causalForAttention,
  } = dispatchParams;

  // 5. Attention (uses raw GPUBuffers)

  // Debug: log scale on layer 0
  if (layerIdx === 0 && isPrefill) {
    trace.attn(layerIdx, `Attention scale=${attnScale.toFixed(6)}, queryPreAttnScalar=${queryPreAttnScalar ?? 'undefined'}, headDim=${headDim}`);
  }

  // Kernel step debug: KV cache state after resolve
  if (isKernelDebugEnabled(layerIdx)) {
    trace.kv(layerIdx, `KV resolved: kvLen=${kvState.kvLenForAttention}, layout=${kvState.kvLayout}`);
  }

  recordAttentionInputs(state, buildAttentionInputsData(
    dispatchConfig, input, normed, kvState, dispatchParams,
    { useF16Activations, matmulOutputDtype },
    usedFusedQKV, qTensor, kTensor, vTensor,
  ));

  const attentionKernelRunners = {
    bdpa: async () => {
      const BDPA_MAX_HEAD_DIM = 256;
      const BDPA_MAX_KV_LEN = 2048;
      if (numTokens !== 1) {
        throw new Error(`BDPA attention supports decode-only seqLen=1; got seqLen=${numTokens}.`);
      }
      if (headDim > BDPA_MAX_HEAD_DIM) {
        throw new Error(
          `BDPA attention kernel supports headDim <= ${BDPA_MAX_HEAD_DIM}; got headDim=${headDim}.`
        );
      }
      if (kvState.kvLenForAttention > BDPA_MAX_KV_LEN) {
        throw new Error(
          `BDPA attention kernel supports kvLen <= ${BDPA_MAX_KV_LEN}; got kvLen=${kvState.kvLenForAttention}.`
        );
      }
      const basisKDtype = 'f16';
      const basisVDtype = 'f16';
      const basisCount = Math.max(1, kvState.bdpaBasisCount);
      const basisKTensor = createTensor(kvState.bdpaBasisK, basisKDtype, [basisCount, numKVHeads * headDim], 'bdpa_basis_k');
      const basisVTensor = createTensor(kvState.bdpaBasisV, basisVDtype, [basisCount, numKVHeads * headDim], 'bdpa_basis_v');
      let qForBDPA = qTensor;
      if (qForBDPA.dtype !== 'f16') {
        qForBDPA = await castF32ToF16(qTensor);
      }
      const output = await runAttentionBDPA(qForBDPA, basisKTensor, basisVTensor, kvState.bdpaPagedK, kvState.bdpaPagedV, kvState.bdpaIndex, numHeads, headDim, {
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
      if (qForBDPA !== qTensor) {
        releaseBuffer(qForBDPA.buffer);
      }
      return output;
    },
    tieredQuant: async () => {
      let qForAttention = qTensor;
      let qTemp = null;
      if (kvState.coldQuantMode !== 'none' && qTensor.dtype !== 'f32') {
        qForAttention = await castF16ToF32(qTensor);
        qTemp = qForAttention;
      }
      const cachedHotKTensor = createTensor(kvState.cachedKHot, cachedKDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_K_hot');
      const cachedHotVTensor = createTensor(kvState.cachedVHot, cachedVDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_V_hot');

      if (kvState.coldQuantMode === 'none') {
        throw new Error('Tiered quant attention requires cold quant mode.');
      }
      if (!kvState.coldScalesK || !kvState.coldScalesV) {
        throw new Error('Tiered quant attention requires cold scale buffers.');
      }

      const output = await runAttentionTieredQuant(
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

      if (qTemp) {
        releaseBuffer(qTemp.buffer);
      }
      return output;
    },
    contiguousQuant: async () => {
      let qForAttention = qTensor;
      let qTemp = null;
      if (qTensor.dtype !== 'f32') {
        qForAttention = await castF16ToF32(qTensor);
        qTemp = qForAttention;
      }

      if (!kvState.coldScalesK || !kvState.coldScalesV) {
        throw new Error('Contiguous quant attention requires scale buffers.');
      }
      if (!kvState.rotationMatrixBuffer || !kvState.codebookCentroidsBuffer) {
        throw new Error('Contiguous quant attention requires TurboQuant shared buffers.');
      }

      const output = await runAttentionContiguousQuant(
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

      if (qTemp) {
        releaseBuffer(qTemp.buffer);
      }
      return output;
    },
    tiered: async () => {
      const qForAttention = qTensor;
      const cachedHotKTensor = createTensor(kvState.cachedKHot, cachedKDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_K_hot');
      const cachedHotVTensor = createTensor(kvState.cachedVHot, cachedVDtype, [kvState.hotLen, numKVHeads * headDim], 'cached_V_hot');
      const cachedColdKTensor = createTensor(kvState.cachedKCold, cachedKDtype, [kvState.coldLen, numKVHeads * headDim], 'cached_K_cold');
      const cachedColdVTensor = createTensor(kvState.cachedVCold, cachedVDtype, [kvState.coldLen, numKVHeads * headDim], 'cached_V_cold');
      return runAttentionTiered(qForAttention, cachedHotKTensor, cachedHotVTensor, cachedColdKTensor, cachedColdVTensor, numHeads, headDim, {
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
        const kCasted = cachedKDtype === 'f16' && kTensor.dtype !== 'f16'
          ? await castF32ToF16(kTensor) : kTensor;
        const vCasted = cachedVDtype === 'f16' && vTensor.dtype !== 'f16'
          ? await castF32ToF16(vTensor) : vTensor;
        kForAttn = createTensor(kCasted.buffer, kCasted.dtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_K');
        vForAttn = createTensor(vCasted.buffer, vCasted.dtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_V');
      }
      const result = await runAttention(qTensor, kForAttn, vForAttn, null, numHeads, headDim, {
        seqLen: numTokens,
        kvLen: kvState.kvLenForAttention,
        numKVHeads,
        causal: causalForAttention,
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
      });
      if (prefillFallbackNeedsCast) {
        if (kTensor.dtype !== 'f16') releaseBuffer(kForAttn.buffer);
        if (vTensor.dtype !== 'f16') releaseBuffer(vForAttn.buffer);
      }
      return result;
    },
  };

  const runAttentionKernel = attentionKernelRunners[attentionKernelVariant];
  if (!runAttentionKernel) {
    throw new Error(`Unsupported attention kernel variant "${attentionKernelVariant}" at layer ${layerIdx}`);
  }

  attnOutput = await runAttentionKernel();

  // Trace attention output
  if (kernelTrace.enabled) {
    await traceStep('attention', `L${layerIdx}.attention`, layerIdx, attnOutput.buffer, [numTokens, numHeads * headDim]);
  }

  // Kernel step debug: attention output
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('attention', { layerIdx, label: `seqLen=${numTokens} kvLen=${kvState.kvLenForAttention}` });
    await dumpTokenVector(attnOutput.buffer, 'attn_out', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: attnOutput.dtype,
    });
  }

  // Debug: attention output for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_out', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(attnOutput.buffer, `L${layerIdx} attention output (before o_proj, GPU)`, numTokens, numHeads * headDim);
  }

  attnForProjection = attnOutput;
  if (qGateTensor) {
    attnForProjection = await runSiLU(attnOutput, {
      size: numTokens * numHeads * headDim,
      gate: qGateTensor,
      gateActivation: 'sigmoid',
      inputActivation: 'identity',
      swigluLimit: null,
    });
    releaseBuffer(attnOutput.buffer);
  }

  // 6. Output projection (with optional fused residual for decode)
  
  output = null;
  let residualFused = false;
  let oProjInput = attnForProjection;
  oProjInputTemp = null;
  if (layerWeights.oProj && getWeightBuffer) {
    ({ oProjInput, oProjInputTemp } = await prepareAttentionProjectionInput(
      attnForProjection,
      matmulOutputDtype,
      castF32ToF16
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
    });  // GEMV kernel expects f16 weights

    if (canUseFused && residualTensor) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await runMatmulResidualFused(oProjInput, oProjBuf, residualTensor, {
        N: hiddenSize,
        K: numHeads * headDim,
      });
      residualFused = true;

      if (layerIdx === 0 && !isPrefill) {
        trace.attn(layerIdx, `Using fused o_proj+residual path`);
      }
    } else {
      // STANDARD PATH: o_proj matmul only (residual will be added by layer.ts)
      output = await runMatmul(oProjInput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
        transposeB: 'auto',
        role: 'o_proj',
        layerIdx,
        kernelPath,
        outputDtype: matmulOutputDtype,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!(layerWeights.oProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.oProj)) {
      releaseBuffer(isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
    }

    // Trace output projection
    if (kernelTrace.enabled) {
      await traceStep('matmul', `L${layerIdx}.o_proj${residualFused ? '+residual' : ''}`, layerIdx, output.buffer, [numTokens, hiddenSize]);
    }

    // Kernel step debug: output projection
    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('matmul', { layerIdx, label: residualFused ? 'O_proj+residual' : 'O_proj', M: numTokens, N: hiddenSize, K: numHeads * headDim });
      await dumpTokenVector(output.buffer, 'o_proj_out', {
        layerIdx,
        tokenIdx: Math.max(0, numTokens - 1),
        rowSize: hiddenSize,
        dtype: output.dtype,
      });
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
        undefined,
        { kernelPath }
      );
      if (combined.buffer !== output.buffer) {
        releaseBuffer(output.buffer);
        output = combined;
      }
    }
  }

  if (oProjInputTemp) {
    releaseBuffer(oProjInputTemp.buffer);
  }

  // Debug: o_proj output for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'o_proj', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(output.buffer, `L${layerIdx} attention output (after o_proj, GPU)`, numTokens, hiddenSize);
  }

  finalOutput = output;
  
  const buffersToRelease = [];
  if (output.buffer !== attnForProjection.buffer) {
    buffersToRelease.push(attnForProjection.buffer);
  }

  if (wantsF16Output && output.dtype !== 'f16') {
    const f16Output = await castF32ToF16(output);
    buffersToRelease.push(output.buffer);
    finalOutput = f16Output;
  }

  // Cleanup
  releaseBuffer(qTensor.buffer);
  if (qGateTensor) {
    releaseBuffer(qGateTensor.buffer);
  }
  releaseBuffer(kTensor.buffer);
  releaseBuffer(vTensor.buffer);
  for (const buffer of buffersToRelease) {
    releaseBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
  } catch (error) {
    const released = new Set();
    const releaseOnce = (buffer) => {
      if (!buffer || released.has(buffer)) return;
      released.add(buffer);
      releaseBuffer(buffer);
    };
    if (finalOutput?.buffer && finalOutput.buffer !== output?.buffer) {
      releaseOnce(finalOutput.buffer);
    }
    if (output?.buffer && output.buffer !== attnForProjection?.buffer) {
      releaseOnce(output.buffer);
    }
    if (oProjInputTemp?.buffer) {
      releaseOnce(oProjInputTemp.buffer);
    }
    if (attnForProjection?.buffer && attnForProjection.buffer !== attnOutput?.buffer) {
      releaseOnce(attnForProjection.buffer);
    }
    if (attnOutput?.buffer) {
      releaseOnce(attnOutput.buffer);
    }
    if (qGateTensor?.buffer) {
      releaseOnce(qGateTensor.buffer);
    }
    if (qTensor?.buffer) {
      releaseOnce(qTensor.buffer);
    }
    if (kTensor?.buffer) {
      releaseOnce(kTensor.buffer);
    }
    if (vTensor?.buffer) {
      releaseOnce(vTensor.buffer);
    }
    if (normed?.buffer && normed.buffer !== attentionInput?.buffer) {
      releaseOnce(normed.buffer);
    }
    if (attentionInputTemp && attentionInput?.buffer) {
      releaseOnce(attentionInput.buffer);
    }
    throw error;
  }
}
