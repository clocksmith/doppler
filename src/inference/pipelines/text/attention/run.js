

import { isWeightBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runRoPE,
  runAttention,
  runAttentionTieredQuant,
  runAttentionTiered,
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
import { SlidingWindowKVCache } from '../../../kv-cache.js';

import {
  shouldDebugLayer,
  markStageLogged,
} from './types.js';

const ATTENTION_DTYPE_LOGGED = new Set();

function recordAttentionInputs(state, info) {
  if (!state?.stats || !info) return;
  if (!state.stats.attentionInputs) {
    state.stats.attentionInputs = [];
  }
  const exists = state.stats.attentionInputs.some(
    (entry) => entry.phase === info.phase && entry.layerIdx === info.layerIdx
  );
  if (exists) return;
  state.stats.attentionInputs.push(info);
}

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
  } = config;

  const device = getDevice();

  const wantsF16Output = input.dtype === 'f16';
  const kvCacheFallback = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
  const kvCacheDtype = state.kvCache?.kvDtype ?? kvCacheFallback;
  const allowF16Attention = wantsF16Output && kvCacheDtype === 'f16';
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output && !allowF16Attention) {
    attentionInput = await castF16ToF32(input);
    attentionInputTemp = true;
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
  
  let normed = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');

    // Debug: norm weights for configured layers
    if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'norm_weights', debugFlags) && debugCheckBuffer) {
      await debugCheckBuffer(normWeightBuf, `L${layerIdx} input norm weights (GPU)`, 1, hiddenSize);
    }

    normed = await runRMSNorm(attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);

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
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = attentionInput.dtype === 'f16';
  const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: useF16Activations,
    fallback: attentionInput.dtype,
  });
  
  let qTensor;
  
  let kTensor;
  
  let vTensor;

  // Check for fused QKV path (3->1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
    getLoRAModule(lora, layerIdx, 'k_proj') ||
    getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = selectRuleValue('inference', 'attention', 'useFusedQkv', {
    hasQkvProj: Boolean(layerWeights.qkvProj),
    hasQkvSizes: Boolean(layerWeights.qkvSizes),
    hasLoRA: Boolean(hasLoRA),
  });

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSize = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvTensor = await runMatmul(normed, layerWeights.qkvProj, numTokens, qkvSize, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
      layerIdx,
      outputDtype: matmulOutputDtype,
    });

    // Split fused output into Q, K, V (returns Tensors)
    const { runSplitQKV } = await import('../../../../gpu/kernels/split_qkv.js');
    const split = await runSplitQKV(qkvTensor, {
      numTokens,
      qSize: qSize_,
      kSize: kSize_,
      vSize: vSize_,
    });
    // Already Tensors from runSplitQKV
    qTensor = split.Q;
    kTensor = split.K;
    vTensor = split.V;

    // Release fused buffer
    releaseBuffer(qkvTensor.buffer);

    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `Using fused QKV path: ${qSize_}+${kSize_}+${vSize_}=${qkvSize}`);
    }
  } else {
    // STANDARD PATH: Separate Q/K/V matmuls
    if (layerWeights.qProj && getWeightBuffer) {
      const qProjBuf = getWeightBuffer(layerWeights.qProj, 'q_proj');
      qTensor = await runMatmul(normed, qProjBuf, numTokens, numHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'q_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.qProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.qProj)) {
        releaseBuffer(isWeightBuffer(qProjBuf) ? qProjBuf.buffer : qProjBuf);
      }
    } else {
      const qBuf = acquireBuffer(qSize * 4, undefined, 'Q');
      qTensor = createTensor(qBuf, normed.dtype, [numTokens, numHeads * headDim], 'Q');
    }

    const loraQ = getLoRAModule(lora, layerIdx, 'q_proj');
    if (loraQ && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        qTensor,
        loraQ,
        { M: numTokens, N: numHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined.buffer !== qTensor.buffer) {
        releaseBuffer(qTensor.buffer);
        qTensor = combined;
      }
    }

    if (layerWeights.kProj && getWeightBuffer) {
      const kProjBuf = getWeightBuffer(layerWeights.kProj, 'k_proj');
      kTensor = await runMatmul(normed, kProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'k_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.kProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.kProj)) {
        releaseBuffer(isWeightBuffer(kProjBuf) ? kProjBuf.buffer : kProjBuf);
      }
    } else {
      const kBuf = acquireBuffer(kvSize * 4, undefined, 'K');
      kTensor = createTensor(kBuf, normed.dtype, [numTokens, numKVHeads * headDim], 'K');
    }

    const loraK = getLoRAModule(lora, layerIdx, 'k_proj');
    if (loraK && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        kTensor,
        loraK,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined.buffer !== kTensor.buffer) {
        releaseBuffer(kTensor.buffer);
        kTensor = combined;
      }
    }

    if (layerWeights.vProj && getWeightBuffer) {
      const vProjBuf = getWeightBuffer(layerWeights.vProj, 'v_proj');

      vTensor = await runMatmul(normed, vProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'v_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.vProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.vProj)) {
        releaseBuffer(isWeightBuffer(vProjBuf) ? vProjBuf.buffer : vProjBuf);
      }
    } else {
      const vBuf = acquireBuffer(kvSize * 4, undefined, 'V');
      vTensor = createTensor(vBuf, normed.dtype, [numTokens, numKVHeads * headDim], 'V');
    }

    const loraV = getLoRAModule(lora, layerIdx, 'v_proj');
    if (loraV && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        vTensor,
        loraV,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined.buffer !== vTensor.buffer) {
        releaseBuffer(vTensor.buffer);
        vTensor = combined;
      }
    }
  }

  // Trace Q/K/V projections
  if (kernelTrace.enabled) {
    await traceStep('matmul', `L${layerIdx}.q_proj`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.k_proj`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.v_proj`, layerIdx, vTensor.buffer, [numTokens, numKVHeads * headDim]);
  }

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

  // Note: Some models use RMSNorm with (1+weight) offset formula for Q/K norms.
  // This is controlled by manifest.inference.normalization.rmsNormWeightOffset.
  if (hasQNorm && getNormWeightBuffer && layerWeights.qNorm) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    // Handle both F16 (2 bytes) and F32 (4 bytes) norm weights
    const qElemsF32 = qNormBuf.size / 4;
    const qElemsF16 = qNormBuf.size / 2;
    const qElems = qElemsF32 === headDim ? qElemsF32 : qElemsF16;
    if (qElems === headDim) {
      const qNormedTensor = await runRMSNorm(qTensor, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseBuffer(qTensor.buffer);
      qTensor = qNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(qTensor.buffer, 'Q_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numHeads * headDim,
          dtype: qTensor.dtype,
        });
      }
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  }

  if (hasKNorm && getNormWeightBuffer && layerWeights.kNorm) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    // Handle both F16 (2 bytes) and F32 (4 bytes) norm weights
    const kElemsF32 = kNormBuf.size / 4;
    const kElemsF16 = kNormBuf.size / 2;
    const kElems = kElemsF32 === headDim ? kElemsF32 : kElemsF16;
    if (kElems === headDim) {
      const kNormedTensor = await runRMSNorm(kTensor, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseBuffer(kTensor.buffer);
      kTensor = kNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(kTensor.buffer, 'K_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numKVHeads * headDim,
          dtype: kTensor.dtype,
        });
      }
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  }

  if (normed !== attentionInput) releaseBuffer(normed.buffer);
  if (attentionInputTemp) releaseBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)

  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await runRoPE(qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await runRoPE(kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });

    // Trace RoPE outputs
    if (kernelTrace.enabled) {
      await traceStep('rope', `L${layerIdx}.q_rope`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
      await traceStep('rope', `L${layerIdx}.k_rope`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    }
  }
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
  const totalSeqLen = currentSeqLen + numTokens;

  const hasCache = state.kvCache?.hasGPUCache?.();

  if (hasCache) {
    if (state.kvCache.kvDtype === 'f16') {
      // Use tensor dtype to determine if cast is needed
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await castF32ToF16(kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await castF32ToF16(vTensor);

      await state.kvCache.updateFromGPU(layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens);

      // Only release if we created new buffers
      if (kTensor.dtype !== 'f16') releaseBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') releaseBuffer(vCasted.buffer);
    } else {
      await state.kvCache.updateFromGPU(layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens);
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
      kvPageSize = gpuBuffers.pageSize ?? state.kvCache.pageSize ?? 0;
      if (state.kvCache instanceof SlidingWindowKVCache) {
        kvLayout = 'ring';
      } else if (state.kvCache.layout === 'paged') {
        kvLayout = 'paged';
      }
    }

    // Kernel step debug: KV cache state after update
    if (isKernelDebugEnabled(layerIdx)) {
      trace.kv(layerIdx, `KV cache updated: kvLen=${kvLenForAttention}, startPos=${currentSeqLen}, numTokens=${numTokens}`);
      await dumpKVCache(( (state.kvCache)), layerIdx);
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
  if (kvLayout !== 'tiered') {
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

  // 5. Attention (uses raw GPUBuffers)
  // query_pre_attn_scalar is used as: scale = scalar^(-0.5) = 1/sqrt(scalar)
  // When scalar equals headDim (e.g., 256): scale = 1/sqrt(256) = 1/16 (standard head_dim scaling)
  const attnScale = queryPreAttnScalar ? 1.0 / Math.sqrt(queryPreAttnScalar) : 1.0 / Math.sqrt(headDim);
  // Debug: log scale on layer 0
  if (layerIdx === 0 && isPrefill) {
    trace.attn(layerIdx, `Attention scale=${attnScale.toFixed(6)}, queryPreAttnScalar=${queryPreAttnScalar ?? 'undefined'}, headDim=${headDim}`);
  }
  // Wrap cached K/V in Tensors (dtype from cache or input tensor)
  const cachedKDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: kTensor.dtype,
  });
  const cachedVDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: vTensor.dtype,
  });
  const cachedKTensor = kvLayout === 'tiered'
    ? null
    : createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = kvLayout === 'tiered'
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
    useFusedQKV,
    kvStart,
    kvLayout,
    kvPageSize: kvLayout === 'tiered' ? (coldPageSize || null) : (kvPageSize || null),
    hotLen: kvLayout === 'tiered' ? hotLen : null,
    coldLen: kvLayout === 'tiered' ? coldLen : null,
    hotWindow: kvLayout === 'tiered' ? hotWindow : null,
    hotStart: kvLayout === 'tiered' ? hotStart : null,
    coldQuantMode: kvLayout === 'tiered' ? coldQuantMode : null,
  });

  let attnOutput;
  if (kvLayout === 'tiered') {
    let qForAttention = qTensor;
    let qTemp = null;
    if (coldQuantMode !== 'none' && qTensor.dtype !== 'f32') {
      qForAttention = await castF16ToF32(qTensor);
      qTemp = qForAttention;
    }
    const cachedHotKTensor = createTensor(cachedKHot, cachedKDtype, [hotLen, numKVHeads * headDim], 'cached_K_hot');
    const cachedHotVTensor = createTensor(cachedVHot, cachedVDtype, [hotLen, numKVHeads * headDim], 'cached_V_hot');
    if (coldQuantMode !== 'none') {
      if (!coldScalesK || !coldScalesV) {
        throw new Error('Tiered quant attention requires cold scale buffers.');
      }
      attnOutput = await runAttentionTieredQuant(qForAttention, cachedHotKTensor, cachedHotVTensor, cachedKCold, cachedVCold, coldScalesK, coldScalesV, numHeads, headDim, {
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
      attnOutput = await runAttentionTiered(qForAttention, cachedHotKTensor, cachedHotVTensor, cachedColdKTensor, cachedColdVTensor, numHeads, headDim, {
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
    if (qTemp) {
      releaseBuffer(qTemp.buffer);
    }
  } else {
    attnOutput = await runAttention(qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
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
      kvLayout,
      kvPageTable,
      kvPageSize,
    });
  }

  // Trace attention output
  if (kernelTrace.enabled) {
    await traceStep('attention', `L${layerIdx}.attention`, layerIdx, attnOutput.buffer, [numTokens, numHeads * headDim]);
  }

  // Kernel step debug: attention output
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('attention', { layerIdx, label: `seqLen=${numTokens} kvLen=${kvLenForAttention}` });
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

  // 6. Output projection (with optional fused residual for decode)
  
  let output;
  let residualFused = false;
  let oProjInput = attnOutput;
  let oProjInputTemp = null;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    if (matmulOutputDtype === 'f16' && attnOutput.dtype !== 'f16') {
      oProjInput = await castF32ToF16(attnOutput);
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
        getWeightBuffer
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

  let finalOutput = output;
  
  const buffersToRelease = [];
  if (output.buffer !== attnOutput.buffer) {
    buffersToRelease.push(attnOutput.buffer);
  }

  if (wantsF16Output && output.dtype !== 'f16') {
    const f16Output = await castF32ToF16(output);
    buffersToRelease.push(output.buffer);
    finalOutput = f16Output;
  }

  // Cleanup
  releaseBuffer(qTensor.buffer);
  releaseBuffer(kTensor.buffer);
  releaseBuffer(vTensor.buffer);
  for (const buffer of buffersToRelease) {
    releaseBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
}
