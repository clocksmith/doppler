

import { log, trace } from '../../../debug/index.js';
import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { createTensor } from '../../../gpu/tensor.js';
import { recordScale, runScale } from '../../../gpu/kernel-selector.js';
import {
  doAttention, doRMSNorm, doSandwichRMSNormPair, doResidualAdd, doMatmul, doGeLU,
  doConv,
  doCast,
  releaseOrTrack
} from './ops.js';
import {
  processFFNWithSandwichNorm,
  processFFNStandard
} from './ffn/index.js';
import { getWeightBuffer, getNormWeightBuffer } from './weights.js';
import { logLayer, logAttn, getBufferStats, isKernelDebugEnabled, dumpTokenVector, logKernelStep, shouldDebugLayerOutput } from './debug-utils/index.js';
import { runProbes } from './probes.js';
import { recordCheckFiniteness } from '../../../gpu/kernels/check-finiteness.js';
import { RMSNORM_PAIR_CACHE_LIMIT } from '../../../gpu/kernel-selector.js';
import { shouldRunFinitenessGuard } from './finiteness-policy.js';
import { runLinearAttentionLayer } from './linear-attention.js';
import { validateAttnConfig } from './attention/attn-config.js';
import { createPerLayerInputTensor, resolveDensePleProjectionWeight } from './per-layer-inputs.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { processLayerPlanGPU } from './layer-plan-gpu.js';
import { isRoPEDisabledForLayer } from './attention/heterogeneous-contract.js';
import { postNormContractMatchesBase } from './normalization-contract.js';
import {
  applyPerLayerInputBlock,
  debugLayerTensor,
  hasPerLayerInputBlock,
  isConvLayerType,
  normalizeLayerType,
  resolveAttentionKVSharing,
  shouldUsePostFfnNextInputRMSNormPairFusion,
  shouldUseStandardPostFfnNextInputRMSNormPairFusion,
} from './layer-input-execution.js';
import {
  applyLayerScalar,
  getConvLayerState,
  isSlidingLayerType,
  resolveActivationDtype,
  resolveAttentionFrequencyBaseDim,
  resolveAttentionHeadDim,
  resolveAttentionNumKVHeads,
  resolveAttentionRotaryDim,
  resolveLayerScalarValue,
} from './layer-execution-contract.js';
export {
  applyPerLayerInputBlock,
  hasPerLayerInputBlock,
  isMoELayer,
  resolveAttentionKVSharing,
} from './layer-input-execution.js';
export {
  applyLayerScalar,
  getConvLayerState,
  isSlidingLayerType,
  resolveActivationDtype,
  resolveAttentionFrequencyBaseDim,
  resolveAttentionHeadDim,
  resolveAttentionNumKVHeads,
  resolveAttentionRotaryDim,
  resolveLayerScalarValue,
} from './layer-execution-contract.js';
// ============================================================================
// Architecture Detection
// ============================================================================

export function detectSandwichNorm(config) {
  const hasPreFeedforwardNorm = config?.preFeedforwardNorm === true;
  const hasPostFeedforwardNorm = config?.postFeedforwardNorm === true;
  const hasPostAttentionNorm = config?.postAttentionNorm === true;

  return {
    useSandwichNorm: hasPreFeedforwardNorm || hasPostFeedforwardNorm,
    hasPreFeedforwardNorm,
    hasPostFeedforwardNorm,
    hasPostAttentionNorm,
  };
}

function shouldUseSandwichRMSNormPairFusion({
  context,
  sandwichNorm,
  layerWeights,
  numTokens,
  hiddenSize,
  attnOutput,
  inputTensor,
}) {
  if (context.config?.normalizationType === 'layernorm' || !postNormContractMatchesBase(context.config)) return false;
  if (context.useSandwichRMSNormPairFusion !== true) {
    return false;
  }
  if (numTokens !== 1) {
    return false;
  }
  if (!sandwichNorm.useSandwichNorm || !sandwichNorm.hasPostAttentionNorm || !sandwichNorm.hasPreFeedforwardNorm) {
    return false;
  }
  if (!layerWeights?.postAttentionNorm || !layerWeights?.preFeedforwardNorm) {
    return false;
  }
  if (attnOutput?.dtype !== 'f32' || (inputTensor && inputTensor.dtype !== 'f32')) {
    throw new Error(
      'useSandwichRMSNormPairFusion requires f32 attention and residual tensors ' +
      `(attn=${String(attnOutput?.dtype)}, residual=${String(inputTensor?.dtype)}).`
    );
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0 || hiddenSize > RMSNORM_PAIR_CACHE_LIMIT) {
    throw new Error(
      `useSandwichRMSNormPairFusion requires hiddenSize in 1..${RMSNORM_PAIR_CACHE_LIMIT}; got ${String(hiddenSize)}.`
    );
  }
  return true;
}

function releasePrecomputedInputNorm(context, recorder) {
  const precomputed = context.__precomputedInputNorm ?? null;
  context.__precomputedInputNorm = null;
  const buffer = precomputed?.tensor?.buffer ?? null;
  if (buffer) {
    releaseOrTrack(recorder, buffer, context.decodeBuffers);
  }
}

function takePrecomputedInputNorm(context, layerIdx, recorder) {
  const precomputed = context.__precomputedInputNorm ?? null;
  if (!precomputed) {
    return null;
  }
  context.__precomputedInputNorm = null;
  if (precomputed.layerIdx !== layerIdx) {
    const buffer = precomputed?.tensor?.buffer ?? null;
    if (buffer) {
      releaseOrTrack(recorder, buffer, context.decodeBuffers);
    }
    throw new Error(
      `Layer ${layerIdx} received stale precomputed input norm for layer ${String(precomputed.layerIdx)}.`
    );
  }
  return precomputed.tensor;
}

const UNSUPPORTED_LAYER_RUNTIME_SET = new Set(['mamba', 'rwkv']);

function assertSupportedLayerRuntime(layerIdx, config) {
  const modelType = normalizeLayerType(config?.modelType);
  if (UNSUPPORTED_LAYER_RUNTIME_SET.has(modelType)) {
    throw new Error(
      `Unsupported runtime family "${modelType}" for layer ${layerIdx}. ` +
      'Mamba/RWKV execution is fail-closed until implemented.'
    );
  }

  const layerType = normalizeLayerType(config?.layerTypes?.[layerIdx]);
  if (UNSUPPORTED_LAYER_RUNTIME_SET.has(layerType)) {
    throw new Error(
      `Unsupported layer type "${layerType}" at layer ${layerIdx}. ` +
      'Mamba/RWKV execution is fail-closed until implemented.'
    );
  }
}

export function hasConvLayers(layerTypes) {
  if (!Array.isArray(layerTypes)) return false;
  for (let i = 0; i < layerTypes.length; i++) {
    if (isConvLayerType(layerTypes[i])) return true;
  }
  return false;
}

function isLinearLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net';
}

// ============================================================================
// Main Layer Processing
// ============================================================================

export async function processLayer(layerIdx, hiddenStates, numTokens, isPrefill, context) {
  const { config, useGPU } = context;
  const { hiddenSize } = config;
  assertSupportedLayerRuntime(layerIdx, config);

  // Debug routing (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  // Debug: check path being taken for layer 0
  if (context.debug && layerIdx === 0) {
    trace.ffn(0, `routing: useGPU=${useGPU}, isGPUBuffer=${isGpuBufferInstance(hiddenStates)}, constructor=${hiddenStates?.constructor?.name}`);
  }

  // GPU-native path
  if (useGPU && isGpuBufferInstance(hiddenStates)) {
    return processLayerGPU(layerIdx, hiddenStates, numTokens, isPrefill, numTokens * hiddenSize, context);
  }

  // CPU fallback path
  return processLayerCPU(layerIdx, (hiddenStates), numTokens, isPrefill, context);
}

// ============================================================================
// GPU Layer Processing
// ============================================================================

export async function processLayerGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context) {
  // Debug entry (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('No GPU device available');

  assertSupportedLayerRuntime(layerIdx, config);
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps, postNormEps = rmsNormEps, postNormWeightOffset = config.rmsNormWeightOffset } = config;
  const residualBranchScale = Number(config.residualBranchScale);
  if (!Number.isFinite(residualBranchScale) || residualBranchScale <= 0) {
    throw new Error(
      `Layer ${layerIdx} residualBranchScale must be a positive finite number; ` +
      `got "${String(config.residualBranchScale)}".`
    );
  }

  // Determine activation dtype from context (defaults to f32)

  const activationDtype = resolveActivationDtype(context.activationDtype);

  // Wrap input buffer as Tensor for dtype-aware processing
  const inputTensor = createTensor(inputBuffer, activationDtype, [numTokens, hiddenSize], 'layer_input');

  const layerWeights = (weights.get(`layer_${layerIdx}`));
  const sandwichNorm = detectSandwichNorm(config);
  if (sandwichNorm.useSandwichNorm && residualBranchScale !== 1) {
    throw new Error(
      `Layer ${layerIdx} uses sandwich norms with residualBranchScale=${residualBranchScale}. ` +
      'Scaled residual branches for sandwich-norm layers are not implemented.'
    );
  }
  const lastTokenIdx = Math.max(0, numTokens - 1);

  await runProbes('layer_in', inputBuffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: inputTensor.dtype,
  });

  if (context.pipelinePlan) {
    if (residualBranchScale !== 1) {
      throw new Error(
        `Layer ${layerIdx} has residualBranchScale=${residualBranchScale}, but pipelinePlan execution ` +
        'does not implement scaled residual branches.'
      );
    }
    return processLayerPlanGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context, layerWeights, sandwichNorm);
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    logKernelStep('layer', { layerIdx, label: `seqLen=${numTokens} prefill=${isPrefill}` });
    await dumpTokenVector(inputBuffer, 'layer_in', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: activationDtype,
    });
  }

  // 1. Layer mixer (attention or conv)
  const layerType = config.layerTypes?.[layerIdx];
  const isConvLayer = isConvLayerType(layerType);
  const isLinearLayer = isLinearLayerType(layerType);
  const isLocalLayer = isSlidingLayerType(layerType);

  // Debug: log RoPE selection for first few layers
  if (context.debug && layerIdx < 3) {
    trace.attn(layerIdx, `Layer routing: layerType=${layerType}, isConv=${isConvLayer}, isLinear=${isLinearLayer}, isLocal=${isLocalLayer}, hasLocalCos=${!!context.ropeLocalCos}, hasLocalSin=${!!context.ropeLocalSin}`);
  }

  let attnOutput;
  let residualFused = false;
  let postAttn = null;
  let fusedResidualForFFN = null;
  try {
  if (isConvLayer) {
    const convInProj = layerWeights?.convInProj ?? null;
    const convOutProj = layerWeights?.convOutProj ?? null;
    if (!convInProj || !convOutProj) {
      throw new Error(
        `Missing conv weights for L${layerIdx}. Expected conv.in_proj.weight and conv.out_proj.weight.`
      );
    }
    const convKernel = layerWeights?.convKernel ?? null;
    // Apply input norm (operator_norm) before conv mixer — matches HF Lfm2 forward pass
    let normedTensor = inputTensor;
    const inputNormWeight = layerWeights?.inputNorm ?? null;
    if (inputNormWeight) {
      const normWeightBuf = getNormWeightBuffer(inputNormWeight, `L${layerIdx}.conv_input_norm`);
      normedTensor = await doRMSNorm(inputTensor, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
        label: `L${layerIdx}.conv_input_norm`,
        layerIdx,
      }, recorder);
      if (!isGpuBufferInstance(inputNormWeight) && !isWeightBuffer(inputNormWeight)) releaseOrTrack(recorder, normWeightBuf);
    }
    attnOutput = await doConv(
      normedTensor,
      getWeightBuffer(convInProj, `L${layerIdx}.conv_in_proj`),
      convKernel ? getWeightBuffer(convKernel, `L${layerIdx}.conv_kernel`) : null,
      getWeightBuffer(convOutProj, `L${layerIdx}.conv_out_proj`),
      {
        numTokens,
        hiddenSize,
        layerIdx,
        label: `L${layerIdx}.conv`,
        swigluLimit: config.swigluLimit,
        kernelPath: context.kernelPath ?? null,
        convState: getConvLayerState(context.convLayerStates, layerIdx),
      },
      recorder
    );
    if (normedTensor !== inputTensor) {
      releaseOrTrack(recorder, normedTensor.buffer);
    }
  } else if (isLinearLayer) {
    attnOutput = await runLinearAttentionLayer(inputTensor, layerWeights ?? null, {
      layerIdx,
      numTokens,
      hiddenSize,
      config,
      currentSeqLen: context.currentSeqLen,
      activationDtype,
      kernelPath: context.kernelPath ?? null,
      linearRuntime: context.linearAttentionRuntime ?? null,
      getWeightBuffer: (weight, label) => getWeightBuffer(weight, label),
      getNormWeightBuffer: (weight, label) => getNormWeightBuffer(weight, label, weightConfig, debugFlags),
      precomputedInputNorm: takePrecomputedInputNorm(context, layerIdx, recorder),
      debugProbes: context.debugProbes,
      operatorDiagnostics: context.operatorDiagnostics,
      recorder: recorder ?? null,
    });
  } else {
    let attentionNumHeads = numHeads;
    let attentionHeadDim = resolveAttentionHeadDim(config, layerType);
    let attentionNumKVHeads = resolveAttentionNumKVHeads(config, layerType, layerWeights, attentionHeadDim);
    const disableRoPE = isRoPEDisabledForLayer(config, layerIdx);
    let queryKeyNorm = config.queryKeyNorm === true;
    const diffusionGemmaDecoder = context.diffusionGemmaDecoder === true;
    if (queryKeyNorm && Array.isArray(config.queryKeyNormLayers)) {
      queryKeyNorm = config.queryKeyNormLayers.includes(layerIdx);
    }
    const { sharedKVSourceLayerIdx, storeSharedKV } = resolveAttentionKVSharing(config, layerIdx, layerType);

    const attnConfig = {
      layerIdx,
      numTokens,
      isPrefill,
      numHeads: attentionNumHeads,
      numKVHeads: attentionNumKVHeads,
      headDim: attentionHeadDim,
      hiddenSize,
      rmsNormEps,
      currentSeqLen: context.currentSeqLen,
      activationDtype,
      slidingWindow: config.slidingWindow,
      layerType,
      residualTensor: (numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm) && residualBranchScale === 1)
        ? inputTensor
        : null,
      attnSoftcap: config.attnLogitSoftcapping === null ? 0 : config.attnLogitSoftcapping,
      queryPreAttnScalar: config.queryPreAttnScalar,
      queryScale: config.queryScale,
      queryKeyNorm,
      queryKeyNormType: config.queryKeyNormType,
      queryKeyNormAxis: config.queryKeyNormAxis,
      queryKeyNormWeightLayers: config.queryKeyNormWeightLayers,
      valueNorm: config.valueNorm,
      attentionOutputGate: config.attentionOutputGate,
      outputGateType: config.outputGateType ?? null,
      causalAttention: diffusionGemmaDecoder ? false : config.causalAttention,
      multimodalBidirectionalSpan: !diffusionGemmaDecoder && isSlidingLayerType(layerType)
        ? (context.multimodalBidirectionalSpan ?? null)
        : null,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
      normalizationType: config.normalizationType,
      ropeRotaryDim: resolveAttentionRotaryDim(config, layerType),
      ropeFrequencyBaseDim: resolveAttentionFrequencyBaseDim(config, layerType),
      ropeInterleaved: config.ropeInterleaved,
      tokenIds: context.currentTokenIds ?? null,
      kernelPath: context.kernelPath ?? null,
      sessionSettings: config.sessionSettings ?? null,
      disableRoPE,
      sharedKVSourceLayerIdx,
      storeSharedKV,
      diffusionGemmaDecoder,
    };

    validateAttnConfig(attnConfig, `L${layerIdx}`);
    attnConfig.precomputedInputNorm = takePrecomputedInputNorm(context, layerIdx, recorder);

    const attnState = {
      ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
        ? (context.ropeLocalCos)
        : (ropeFreqsCos),
      ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
        ? (context.ropeLocalSin)
        : (ropeFreqsSin),
      sharedAttentionState: context.sharedAttentionState ?? null,
      kvCache: ((kvCache)),
      stats: context.stats,
      debugProbes: context.debugProbes,
      operatorDiagnostics: context.operatorDiagnostics,
      linearRuntime: context.linearAttentionRuntime ?? null,
      resolvedRuntimeSession: context.resolvedRuntimeSession,
      observationContext: context.observationContext ?? null,
      executionPolicies: context.executionPolicies ?? null,
      skipKVCacheWrites: context.skipKVCacheWrites === true,
    };

    const attnResult = await doAttention(
      inputTensor,
      layerWeights ?? null,
      attnConfig,
      attnState,
      context.debug,
      { debugLayers: context.debugLayers },
      (weight, label) => getWeightBuffer(weight, label),
      (weight, label) => getNormWeightBuffer(weight, label, weightConfig, debugFlags),
      context.debugCheckBuffer,
      recorder,
      context.lora
    );
    attnOutput = attnResult.output;
    residualFused = attnResult.residualFused;
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(attnOutput.buffer, 'attn_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: attnOutput.dtype,
    });
  }

  // Debug: trace attn output
  if (context.debug) {
    const stats = await getBufferStats(attnOutput.buffer);
    if (stats) logAttn(layerIdx, isPrefill, { numTokens, kvLen: context.currentSeqLen + (isPrefill ? numTokens : 1), maxAbsOut: stats.maxAbs });

    trace.attn(layerIdx, `attnOutput type check: isGPU=${isGpuBufferInstance(attnOutput.buffer)}, type=${typeof attnOutput.buffer}, constructor=${attnOutput.buffer?.constructor?.name}, isPrefill=${isPrefill}`);
    if (shouldDebugLayerOutput(layerIdx, context.debugLayers) && isGpuBufferInstance(attnOutput.buffer) && !recorder) {
      if (allowReadback(`layer.attn-out.${layerIdx}`)) {
        try {
          const sampleSize = Math.min(128, attnOutput.buffer.size);
          const data = new Float32Array(await readBuffer(attnOutput.buffer, sampleSize));
          let maxAbs = 0;
          for (let i = 0; i < data.length; i++) {
            const abs = Math.abs(data[i]);
            if (abs > maxAbs) maxAbs = abs;
          }
          const nonZero = Array.from(data).filter(x => x !== 0).length;
          trace.attn(layerIdx, `ATTN_OUT: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) {
          trace.attn(layerIdx, `ATTN_OUT error: ${e}`);
        }
      }
    } else if (shouldDebugLayerOutput(layerIdx, context.debugLayers) && isGpuBufferInstance(attnOutput.buffer) && recorder) {
      trace.attn(layerIdx, `ATTN_OUT: (skipped - using batched recorder, values not available until submit)`);
    }
  }
  await runProbes('attn_out', attnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: attnOutput.dtype,
  });
  if (!residualFused && residualBranchScale !== 1) {
    const rawAttnOutput = attnOutput;
    attnOutput = recorder
      ? await recordScale(recorder, rawAttnOutput, residualBranchScale, { count: size })
      : await runScale(rawAttnOutput, residualBranchScale, { count: size });
    releaseOrTrack(recorder, rawAttnOutput.buffer, context.decodeBuffers);
  }

  // 2. Handle residual connection based on architecture

  let precomputedFfnInput = null;
  if (residualFused) {
    postAttn = attnOutput;
    if (shouldUseSandwichRMSNormPairFusion({
      context,
      sandwichNorm,
      layerWeights,
      numTokens,
      hiddenSize,
      attnOutput,
      inputTensor: null,
    })) {
      const postNormWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
      const preNormWeightBuf = getNormWeightBuffer(layerWeights.preFeedforwardNorm, 'pre_feedforward_norm', weightConfig, debugFlags);
      const pair = await doSandwichRMSNormPair(attnOutput, null, postNormWeightBuf, preNormWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_pre_ffn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
      postAttn = pair.postAttn;
      precomputedFfnInput = pair.ffnInput;
      if (!isGpuBufferInstance(layerWeights.postAttentionNorm) && !isWeightBuffer(layerWeights.postAttentionNorm)) releaseOrTrack(recorder, postNormWeightBuf);
      if (!isGpuBufferInstance(layerWeights.preFeedforwardNorm) && !isWeightBuffer(layerWeights.preFeedforwardNorm)) releaseOrTrack(recorder, preNormWeightBuf);
      if (recorder) {
        recorder.trackTemporaryBuffer(attnOutput.buffer);
      } else {
        releaseBuffer(attnOutput.buffer);
      }
    } else if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
      const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, postNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: postNormWeightOffset,
      }, recorder);
      if (!isGpuBufferInstance(layerWeights.postAttentionNorm) && !isWeightBuffer(layerWeights.postAttentionNorm)) releaseOrTrack(recorder, normWeightBuf);
      if (recorder) {
        recorder.trackTemporaryBuffer(attnOutput.buffer);
      } else {
        releaseBuffer(attnOutput.buffer);
      }
    }
  } else if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
    if (shouldUseSandwichRMSNormPairFusion({
      context,
      sandwichNorm,
      layerWeights,
      numTokens,
      hiddenSize,
      attnOutput,
      inputTensor,
    })) {
      const preNormWeightBuf = getNormWeightBuffer(layerWeights.preFeedforwardNorm, 'pre_feedforward_norm', weightConfig, debugFlags);
      const pair = await doSandwichRMSNormPair(attnOutput, inputTensor, normWeightBuf, preNormWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_pre_ffn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
      postAttn = pair.postAttn;
      precomputedFfnInput = pair.ffnInput;
      if (!isGpuBufferInstance(layerWeights.preFeedforwardNorm) && !isWeightBuffer(layerWeights.preFeedforwardNorm)) releaseOrTrack(recorder, preNormWeightBuf);
    } else if (attnOutput.dtype === inputTensor.dtype) {
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, postNormEps, {
        batchSize: numTokens,
        hiddenSize,
        residual: inputTensor,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: postNormWeightOffset,
      }, recorder);
    } else {
      const normalizedAttn = await doRMSNorm(attnOutput, normWeightBuf, postNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: postNormWeightOffset,
      }, recorder);
      postAttn = await doResidualAdd(normalizedAttn, inputTensor, size, recorder, {
        label: `L${layerIdx}.post_attn_residual`,
        layerIdx,
        executionPolicies: context.executionPolicies ?? null,
      });
      releaseOrTrack(recorder, normalizedAttn.buffer, context.decodeBuffers);
    }

    if (!isGpuBufferInstance(layerWeights.postAttentionNorm) && !isWeightBuffer(layerWeights.postAttentionNorm)) releaseOrTrack(recorder, normWeightBuf);
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  } else if (layerWeights?.postAttnNorm) {
    // Fused path: defer residual add into processFFNStandard's rmsnorm (PRE_RESIDUAL).
    // Saves one residual_add dispatch per layer. The rmsnorm computes
    // rmsnorm(attnOutput + inputTensor) and writes the pre-norm sum for downstream use.
    postAttn = attnOutput;
    fusedResidualForFFN = inputTensor;
  } else {
    postAttn = await doResidualAdd(attnOutput, inputTensor, size, recorder, {
      label: `L${layerIdx}.post_attn_residual`,
      layerIdx,
      executionPolicies: context.executionPolicies ?? null,
    });
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(postAttn.buffer, 'x_after_attn', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: postAttn.dtype,
    });
  }

  await runProbes('post_attn', postAttn.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: postAttn.dtype,
  });

  // 3. Feed-forward network

  let outputTensor;
  const layerScalar = resolveLayerScalarValue(layerWeights?.layerScalar ?? null);
  const requestFfnLayerScalarFusion = layerScalar !== 1
    && !hasPerLayerInputBlock(config);
  const nextLayerIdx = layerIdx + 1;
  const nextLayerWeights = nextLayerIdx < config.numLayers
    ? weights.get(`layer_${nextLayerIdx}`)
    : null;
  const usePostFfnNextInputNormPair = shouldUsePostFfnNextInputRMSNormPairFusion({
    context,
    config,
    sandwichNorm,
    layerIdx,
    layerWeights,
    nextLayerWeights,
    numTokens,
    hiddenSize,
    activationDtype,
    layerScalar,
  });
  const useStandardPostFfnNextInputNormPair = shouldUseStandardPostFfnNextInputRMSNormPairFusion({
    context,
    config,
    sandwichNorm,
    layerIdx,
    nextLayerWeights,
    numTokens,
    hiddenSize,
    activationDtype,
    layerScalar,
    residualBranchScale,
  });
  let layerScalarFused = false;
  if (sandwichNorm.useSandwichNorm) {
    context.__postFfnNextInputNorm = usePostFfnNextInputNormPair
      ? { layerIdx: nextLayerIdx, weight: nextLayerWeights.inputNorm }
      : null;
    try {
      outputTensor = await processFFNWithSandwichNorm(
        layerIdx,
        postAttn,
        numTokens,
        size,
        context,
        layerWeights,
        sandwichNorm,
        requestFfnLayerScalarFusion ? layerScalar : 1,
        precomputedFfnInput
      );
    } finally {
      context.__postFfnNextInputNorm = null;
    }
    layerScalarFused = context.__layerScalarFusedFired === true;
    context.__layerScalarFusedFired = false;
  } else {
    outputTensor = await processFFNStandard(
      layerIdx,
      postAttn,
      numTokens,
      size,
      context,
      layerWeights,
      fusedResidualForFFN,
      requestFfnLayerScalarFusion ? layerScalar : 1,
      residualBranchScale,
      useStandardPostFfnNextInputNormPair
        ? { layerIdx: nextLayerIdx, weight: nextLayerWeights.inputNorm }
        : null
    );
    layerScalarFused = context.__layerScalarFusedFired === true;
    context.__layerScalarFusedFired = false;
  }

  // Keep activation dtype consistent across layers. Some FFN paths can emit f32
  // tensors even when the execution plan is f16; leaving that unnormalized causes
  // downstream kernels to decode the buffer with the wrong dtype contract. Apply
  // the cast BEFORE the PLE block so per_layer_input_gate's matmul inherits a
  // consistent outputDtype (gelu_f16 reads gate as array<f16>; binding an f32
  // buffer there reinterprets bytes and produces NaN/garbage).
  if (outputTensor.dtype !== activationDtype) {
    const widened = outputTensor;
    outputTensor = await doCast(widened, activationDtype, recorder);
    releaseOrTrack(recorder, widened.buffer, context.decodeBuffers);
  }

  if (hasPerLayerInputBlock(config)) {
    const outputWithPerLayerInput = await applyPerLayerInputBlock(
      layerIdx,
      outputTensor,
      numTokens,
      size,
      context,
      layerWeights
    );
    if (outputWithPerLayerInput.buffer !== outputTensor.buffer) {
      releaseOrTrack(recorder, outputTensor.buffer, context.decodeBuffers);
    }
    outputTensor = outputWithPerLayerInput;
  }

  // Re-normalize after PLE: the residual add inside applyPerLayerInputBlock
  // may emit f32 even when the layer's activation dtype is f16, which would
  // misroute the next layer's f16 input bindings.
  if (outputTensor.dtype !== activationDtype) {
    const widened = outputTensor;
    outputTensor = await doCast(widened, activationDtype, recorder);
    releaseOrTrack(recorder, widened.buffer, context.decodeBuffers);
  }

  let finalOutput = outputTensor;
  if (!layerScalarFused) {
    const scaledOutput = await applyLayerScalar(layerIdx, finalOutput, size, context, layerWeights);
    if (scaledOutput.buffer !== finalOutput.buffer) {
      releaseOrTrack(recorder, finalOutput.buffer, context.decodeBuffers);
      finalOutput = scaledOutput;
    }
  }
  await debugLayerTensor(context, layerIdx, 'final layer output', finalOutput, numTokens, hiddenSize);
  await runProbes('layer_out', finalOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: finalOutput.dtype,
  });

  // Early-stop check for F16 NaN/Infinity bounds
  const computeConfig = context.runtimeComputeConfig ?? null;
  const shouldCheckFiniteness = context.finitenessGuardEnabled !== undefined
    ? context.finitenessGuardEnabled
    : shouldRunFinitenessGuard(context.activationDtype, computeConfig);
  if (context.finitenessBuffer && context.activationDtype === 'f16' && shouldCheckFiniteness) {
    recordCheckFiniteness(
      recorder,
      finalOutput.buffer,
      size,
      context.finitenessBuffer,
      layerIdx,
      context.step,
      context.finitenessAbsThreshold
    );
  }

  return finalOutput.buffer;
  } catch (error) {
    // Release any intermediate buffers allocated during step execution
    const released = new Set();
    const releaseOnce = (buf) => {
      if (!buf || released.has(buf) || buf === inputBuffer) return;
      released.add(buf);
      releaseOrTrack(recorder, buf);
    };
    releasePrecomputedInputNorm(context, recorder);
    if (postAttn?.buffer) releaseOnce(postAttn.buffer);
    if (attnOutput?.buffer && attnOutput.buffer !== postAttn?.buffer) releaseOnce(attnOutput.buffer);
    throw error;
  }
}

// ============================================================================
// CPU Fallback
// ============================================================================

async function processLayerCPU(layerIdx, hiddenStates, numTokens, isPrefill, context) {
  const { config } = context;
  assertSupportedLayerRuntime(layerIdx, config);
  const { hiddenSize } = config;

  log.warn('Layer', `L${layerIdx} CPU fallback - returning input unchanged`);
  return new Float32Array(hiddenStates);
}
