

import { isWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import {
  recordMatmul,
  recordRMSNorm,
  recordRoPE,
  recordAttention,
  recordAttentionTiered,
  recordAttentionTieredQuant,
  recordAttentionBDPA,
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
import { SlidingWindowKVCache } from '../../../kv-cache.js';
import {
  recordAttentionInputs,
  resolveAttentionProjectionOutputDtype,
  projectAttentionQKV,
  applyAttentionQKNorm,
} from './projections.js';

import { releaseOrTrack, shouldDebugLayer } from './types.js';

const ATTENTION_DTYPE_LOGGED = new Set();


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
  } = config;

  const wantsF16Output = input.dtype === 'f16';
  const kvCacheFallback = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
  const kvCacheDtype = state.kvCache?.kvDtype ?? kvCacheFallback;
  const allowF16Attention = wantsF16Output && kvCacheDtype === 'f16';
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output && !allowF16Attention) {
    attentionInput = await recordCastF16ToF32(recorder, input);
    attentionInputTemp = true;
  }

  if (!layerWeights) {
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const outputDtype = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
    const output = createTensor(outputBuf, outputDtype, [numTokens, hiddenSize], 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm

  let normed = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
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

  // 2. Q/K/V projections
  const useF16Activations = attentionInput.dtype === 'f16';
  const matmulOutputDtype = resolveAttentionProjectionOutputDtype(attentionInput.dtype);
  let { qTensor, kTensor, vTensor, usedFusedQKV } = await projectAttentionQKV({
    recorder,
    normed,
    layerWeights,
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    layerIdx,
    matmulOutputDtype,
    getWeightBuffer,
    lora,
    releaseTemporary: (buffer) => releaseOrTrack(recorder, buffer),
  });

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
  }));

  if (normed !== attentionInput) releaseOrTrack(recorder, normed.buffer);
  if (attentionInputTemp) recorder.trackTemporaryBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)
  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await recordRoPE(recorder, qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await recordRoPE(recorder, kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)

  let cachedK;

  let cachedV;
  let kvLenForAttention = currentSeqLen + numTokens;
  let causalForAttention = config.causalAttention !== false;
  let startPosForMask = currentSeqLen;
  let kvStart = 0;
  let kvLayout = 'contiguous';
  let kvPageTable = null;
  let kvPageSize = 0;
  let cachedKHot;
  let cachedVHot;
  let cachedKCold;
  let cachedVCold;
  let coldScalesK = null;
  let coldScalesV = null;
  let coldPackedStride = 0;
  let coldQuantMode = 'none';
  let coldLen = 0;
  let hotLen = 0;
  let hotStart = 0;
  let hotWindow = 0;
  let coldPageTable = null;
  let coldPageSize = 0;

  // BDPA specifics
  let bdpaBasisK = null;
  let bdpaBasisV = null;
  let bdpaPagedK = null;
  let bdpaPagedV = null;
  let bdpaIndex = null;
  let bdpaBasisCount = 0;

  const totalSeqLen = currentSeqLen + numTokens;

  const hasCache = state.kvCache?.hasGPUCache?.();

  if (hasCache) {
    // Use recordUpdateFromGPU to record copy operations to the recorder's encoder
    // This ensures K/V buffers are populated before copying (all ops submitted together)
    if (state.kvCache.kvDtype === 'f16') {
      // Use tensor dtype to determine if cast is needed
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await recordCastF32ToF16(recorder, kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await recordCastF32ToF16(recorder, vTensor);

      await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens, tokenIds);

      // Track for cleanup after submit (not release!) - only if we created new buffers
      if (kTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(vCasted.buffer);
    } else {
      await state.kvCache.recordUpdateFromGPU(recorder, layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens, tokenIds);
    }
    const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
    if (gpuBuffers?.layout === 'tiered') {
      cachedKHot = gpuBuffers.hotKeysGPU;
      cachedVHot = gpuBuffers.hotValuesGPU;
      cachedKCold = gpuBuffers.coldKeysGPU;
      cachedVCold = gpuBuffers.coldValuesGPU;
      coldScalesK = gpuBuffers.coldScalesKGPU ?? null;
      coldScalesV = gpuBuffers.coldScalesVGPU ?? null;
      coldPackedStride = gpuBuffers.coldPackedStride ?? 0;
      coldQuantMode = gpuBuffers.coldQuantMode ?? 'none';
      hotLen = gpuBuffers.hotSeqLen ?? 0;
      coldLen = gpuBuffers.coldSeqLen ?? 0;
      hotStart = gpuBuffers.hotStart ?? 0;
      hotWindow = gpuBuffers.hotWindow ?? 0;
      coldPageTable = gpuBuffers.coldPageTableGPU ?? null;
      coldPageSize = gpuBuffers.coldPageSize ?? state.kvCache.coldPageSize ?? 0;
      kvLenForAttention = coldLen + hotLen;
      kvLayout = 'tiered';
    } else {
      cachedK = gpuBuffers.keysGPU;
      cachedV = gpuBuffers.valuesGPU;
      kvLenForAttention = gpuBuffers.seqLen;
      kvPageTable = gpuBuffers.pageTableGPU ?? null;
      if (state.kvCache instanceof SlidingWindowKVCache) {
        kvLayout = 'ring';
      } else if (state.kvCache.layout === 'paged') {
        kvLayout = 'paged';
      } else if (gpuBuffers.layout === 'bdpa') {
        kvLayout = 'bdpa';
        bdpaBasisK = gpuBuffers.basisGPU.k;
        bdpaBasisV = gpuBuffers.basisGPU.v;
        bdpaPagedK = gpuBuffers.pagedGPU.k;
        bdpaPagedV = gpuBuffers.pagedGPU.v;
        bdpaIndex = gpuBuffers.indexGPU;
        bdpaBasisCount = gpuBuffers.numBasisVectors ?? state.kvCache.basisVocabSize;
      }
    }
  } else {
    cachedK = kTensor.buffer;
    cachedV = vTensor.buffer;
    kvLenForAttention = numTokens;
    startPosForMask = 0;
  }

  if (kvLayout === 'tiered' && numTokens > 1) {
    kvLayout = 'contiguous';
    cachedK = kTensor.buffer;
    cachedV = vTensor.buffer;
    kvLenForAttention = numTokens;
    startPosForMask = 0;
    cachedKHot = null;
    cachedVHot = null;
    cachedKCold = null;
    cachedVCold = null;
    coldQuantMode = 'none';
  }

  // Sliding window attention for specific layers
  // The kernel now handles both causal AND sliding window masking together.
  // We no longer need to disable causal masking for sliding layers.
  const hasSlidingWindow = Number.isFinite(slidingWindow) && slidingWindow > 0;
  const hasLayerTypes = Array.isArray(config.layerTypes);
  const isLayerSliding = layerType === 'sliding_attention' || (hasSlidingWindow && !hasLayerTypes);
  const effectiveSlidingWindow = isLayerSliding ? slidingWindow : null;
  const canWindow = hasCache && effectiveSlidingWindow;
  if (kvLayout !== 'tiered' && kvLayout !== 'bdpa') {
    if (canWindow && kvLenForAttention > effectiveSlidingWindow) {
      kvLenForAttention = effectiveSlidingWindow;
    }
    if (hasCache && (kvLayout === 'ring' || (canWindow && kvLenForAttention < totalSeqLen))) {
      kvStart = Math.max(0, totalSeqLen - kvLenForAttention);
    }
  }

  if (kvLenForAttention <= 0) {
    throw new Error(`Invalid kvLen ${kvLenForAttention} at layer ${layerIdx}`);
  }

  // 5. Attention
  // query_pre_attn_scalar is used as: scale = scalar^(-0.5) = 1/sqrt(scalar)
  // When scalar equals headDim (e.g., 256): scale = 1/sqrt(256) = 1/16 (standard head_dim scaling)
  const attnScale = queryPreAttnScalar ? 1.0 / Math.sqrt(queryPreAttnScalar) : 1.0 / Math.sqrt(headDim);

  // Wrap cached K/V in Tensors (dtype from cache or input tensor)
  const cachedKDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: kTensor.dtype,
  });
  const cachedVDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: vTensor.dtype,
  });
  const cachedKTensor = kvLayout === 'tiered' || kvLayout === 'bdpa'
    ? null
    : createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = kvLayout === 'tiered' || kvLayout === 'bdpa'
    ? null
    : createTensor(cachedV, cachedVDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_V');

  recordAttentionInputs(state, {
    phase: isPrefill ? 'prefill' : 'decode',
    layerIdx,
    numTokens,
    kvLen: kvLenForAttention,
    numHeads,
    numKVHeads,
    headDim,
    activationDtype: config.activationDtype ?? null,
    inputDtype: input.dtype,
    normedDtype: normed.dtype,
    useF16Activations,
    matmulOutputDtype,
    kvCacheDtype: state.kvCache?.kvDtype ?? null,
    cachedKDtype,
    cachedVDtype,
    qDtype: qTensor?.dtype ?? null,
    kDtype: kTensor?.dtype ?? null,
    vDtype: vTensor?.dtype ?? null,
    useFusedQKV: usedFusedQKV,
    kvStart,
    kvLayout,
    kvPageSize: kvLayout === 'tiered' ? (coldPageSize || null) : (kvPageSize || null),
    hotLen: kvLayout === 'tiered' ? hotLen : null,
    coldLen: kvLayout === 'tiered' ? coldLen : null,
    hotStart: kvLayout === 'tiered' ? hotStart : null,
  });

  let attnOutput;
  if (kvLayout === 'bdpa') {
    const basisKDtype = 'f16';
    const basisVDtype = 'f16';
    const numBasisVectors = Math.max(1, bdpaBasisCount);
    const basisKTensor = createTensor(bdpaBasisK, basisKDtype, [numBasisVectors, numKVHeads * headDim], 'bdpa_basis_k');
    const basisVTensor = createTensor(bdpaBasisV, basisVDtype, [numBasisVectors, numKVHeads * headDim], 'bdpa_basis_v');

    let qForBDPA = qTensor;
    if (qForBDPA.dtype !== 'f16') {
      qForBDPA = await recordCastF32ToF16(recorder, qTensor);
      recorder.trackTemporaryBuffer(qForBDPA.buffer);
    }

    attnOutput = await recordAttentionBDPA(recorder, qForBDPA, basisKTensor, basisVTensor, bdpaPagedK, bdpaPagedV, bdpaIndex, numHeads, headDim, {
      seqLen: numTokens,
      kvLen: kvLenForAttention,
      numKVHeads,
      causal: causalForAttention,
      startPos: startPosForMask,
      layerIdx,
      slidingWindow: effectiveSlidingWindow,
      attnSoftcap,
      scale: attnScale,
      ropeCos: state.ropeFreqsCos,
      ropeSin: state.ropeFreqsSin,
    });
  } else if (kvLayout === 'tiered') {
    let qForAttention = qTensor;
    if (coldQuantMode !== 'none' && qTensor.dtype !== 'f32') {
      qForAttention = await recordCastF16ToF32(recorder, qTensor);
      recorder.trackTemporaryBuffer(qForAttention.buffer);
    }
    const cachedHotKTensor = createTensor(cachedKHot, cachedKDtype, [hotLen, numKVHeads * headDim], 'cached_K_hot');
    const cachedHotVTensor = createTensor(cachedVHot, cachedVDtype, [hotLen, numKVHeads * headDim], 'cached_V_hot');
    if (coldQuantMode !== 'none') {
      if (!coldScalesK || !coldScalesV) {
        throw new Error('Tiered quant attention requires cold scale buffers.');
      }
      attnOutput = await recordAttentionTieredQuant(recorder, qForAttention, cachedHotKTensor, cachedHotVTensor, cachedKCold, cachedVCold, coldScalesK, coldScalesV, numHeads, headDim, {
        seqLen: numTokens,
        coldLen,
        hotLen,
        numKVHeads,
        causal: causalForAttention,
        startPos: startPosForMask,
        slidingWindow: effectiveSlidingWindow ?? 0,
        attnSoftcap,
        scale: attnScale,
        hotWindow,
        hotStart,
        packedStride: coldPackedStride,
        mode: coldQuantMode,
      });
    } else {
      const cachedColdKTensor = createTensor(cachedKCold, cachedKDtype, [coldLen, numKVHeads * headDim], 'cached_K_cold');
      const cachedColdVTensor = createTensor(cachedVCold, cachedVDtype, [coldLen, numKVHeads * headDim], 'cached_V_cold');
      attnOutput = await recordAttentionTiered(recorder, qForAttention, cachedHotKTensor, cachedHotVTensor, cachedColdKTensor, cachedColdVTensor, numHeads, headDim, {
        seqLen: numTokens,
        coldLen,
        hotLen,
        numKVHeads,
        causal: causalForAttention,
        startPos: startPosForMask,
        slidingWindow: effectiveSlidingWindow ?? 0,
        attnSoftcap,
        scale: attnScale,
        hotWindow,
        hotStart,
        coldPageTable,
        coldPageSize,
        coldLayout: coldPageTable ? 2 : 0,
        hotLayout: hotWindow > 0 ? 1 : 0,
      });
    }
  } else {
    /** @type {'contiguous' | 'ring' | 'paged' | 'bdpa'} */
    const typeCheckedLayout = kvLayout === 'tiered' ? 'contiguous' : kvLayout;

    // We cast to `any` because `recordAttention` typings enforce contiguous/ring/paged. BDPA has its own path now but we need to satisfy TS.
    attnOutput = await recordAttention(recorder, qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
      seqLen: numTokens,
      kvLen: kvLenForAttention,
      numKVHeads,
      causal: causalForAttention,
      startPos: startPosForMask,
      layerIdx,
      slidingWindow: effectiveSlidingWindow,
      attnSoftcap,
      scale: attnScale,
      kvStart,
      kvLayout: /** @type {any} */ (typeCheckedLayout),
      kvPageTable,
      kvPageSize,
    });
  }

  // 6. Output projection (with optional fused residual for decode)

  let output;
  let residualFused = false;
  let oProjInput = attnOutput;
  let oProjInputTemp = null;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    if (matmulOutputDtype === 'f16' && attnOutput.dtype !== 'f16') {
      oProjInput = await recordCastF32ToF16(recorder, attnOutput);
      oProjInputTemp = oProjInput;
    }

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
        outputDtype: matmulOutputDtype,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!(layerWeights.oProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.oProj)) {
      releaseOrTrack(recorder, isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
    }
  } else {
    output = attnOutput;
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
        recorder
      );
      if (combined.buffer !== output.buffer) {
        recorder.trackTemporaryBuffer(output.buffer);
        output = combined;
      }
    }
  }

  let finalOutput = output;

  const buffersToTrack = [];
  if (output.buffer !== attnOutput.buffer) {
    buffersToTrack.push(attnOutput.buffer);
  }
  if (oProjInputTemp && oProjInputTemp.buffer !== attnOutput.buffer) {
    buffersToTrack.push(oProjInputTemp.buffer);
  }
  if (wantsF16Output && output.dtype !== 'f16') {
    const f16Output = await recordCastF32ToF16(recorder, output);
    buffersToTrack.push(output.buffer);
    finalOutput = f16Output;
  }

  // Track intermediate buffers for cleanup after submit (not release!)
  // These buffers are used by recorded operations that haven't executed yet.
  // Releasing them back to the pool would allow reuse before the encoder is submitted,
  // causing data corruption (especially for small decode buffers).
  recorder.trackTemporaryBuffer(qTensor.buffer);
  recorder.trackTemporaryBuffer(kTensor.buffer);
  recorder.trackTemporaryBuffer(vTensor.buffer);
  for (const buffer of buffersToTrack) {
    recorder.trackTemporaryBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
}
