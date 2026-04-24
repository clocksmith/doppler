

import { log, trace } from '../../../debug/index.js';
import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { createTensor } from '../../../gpu/tensor.js';
import { recordScale, runScale } from '../../../gpu/kernel-selector.js';
import {
  doAttention, doRMSNorm, doResidualAdd, doMatmul, doGeLU,
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
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { recordCheckFiniteness } from '../../../gpu/kernels/check-finiteness.js';
import { shouldRunFinitenessGuard } from './finiteness-policy.js';
import { runLinearAttentionLayer } from './linear-attention.js';
import { validateAttnConfig } from './attention/attn-config.js';
import { createPerLayerInputTensor, resolveDensePleProjectionWeight } from './per-layer-inputs.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { processLayerPlanGPU } from './layer-plan-gpu.js';

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


export function isMoELayer(layerIdx, config) {
  if (!config.useMoE) return false;

  // Manifest-first: check layerTypes from config (derived from manifest.inference.layerPattern)
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerIdx < layerTypes.length) {
    return layerTypes[layerIdx] === 'moe';
  }

  // No layerTypes available: assume all layers are MoE
  return true;
}

export function resolveActivationDtype(dtype) {
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype });
}

function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
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

export function getConvLayerState(convLayerStates, layerIdx) {
  if (!convLayerStates) return {};
  return convLayerStates.get(layerIdx) ?? {};
}

export function isSlidingLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding';
}

function isConvLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'conv'
    || normalized === 'convolution'
    || normalized === 'liv_conv'
    || normalized === 'liv_convolution';
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

export function resolveAttentionRotaryDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.ropeLocalRotaryDim ?? config.ropeRotaryDim;
  }
  return config.ropeRotaryDim;
}

export function resolveAttentionFrequencyBaseDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.ropeLocalFrequencyBaseDim ?? resolveAttentionRotaryDim(config, layerType);
  }
  return config.ropeFrequencyBaseDim ?? config.ropeRotaryDim;
}

export function resolveAttentionHeadDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.headDim;
  }
  return config.globalHeadDim ?? config.headDim;
}

function resolveProjectionOutputRows(layerWeight, hiddenSize) {
  if (!layerWeight || !Array.isArray(layerWeight.shape) || layerWeight.shape.length < 2) {
    return null;
  }
  const dim0 = Number(layerWeight.shape[0]);
  const dim1 = Number(layerWeight.shape[1]);
  if (!Number.isFinite(dim0) || !Number.isFinite(dim1)) {
    return null;
  }
  if (dim1 === hiddenSize) {
    return Math.trunc(dim0);
  }
  if (dim0 === hiddenSize) {
    return Math.trunc(dim1);
  }
  return null;
}

export function resolveAttentionNumKVHeads(config, layerType, layerWeights, headDim) {
  const kRows = resolveProjectionOutputRows(layerWeights?.kProj, config.hiddenSize);
  if (kRows != null && Number.isFinite(headDim) && headDim > 0 && kRows % headDim === 0) {
    return kRows / headDim;
  }
  if (!isSlidingLayerType(layerType) && Number.isFinite(config.numGlobalKVHeads) && config.numGlobalKVHeads > 0) {
    return Math.trunc(config.numGlobalKVHeads);
  }
  return config.numKVHeads;
}

export function resolveAttentionKVSharing(config, layerIdx, layerType) {
  const layerTypes = Array.isArray(config?.layerTypes) ? config.layerTypes : null;
  const numKvSharedLayers = Number(config?.numKvSharedLayers ?? 0);
  if (!layerTypes || layerTypes.length === 0 || !Number.isFinite(numKvSharedLayers) || numKvSharedLayers <= 0) {
    return { sharedKVSourceLayerIdx: null, storeSharedKV: false };
  }

  const firstKvSharedLayerIdx = layerTypes.length - Math.trunc(numKvSharedLayers);
  if (firstKvSharedLayerIdx <= 0 || layerIdx < 0 || layerIdx >= layerTypes.length) {
    return { sharedKVSourceLayerIdx: null, storeSharedKV: false };
  }

  const normalizedLayerType = normalizeLayerType(layerType);
  if (!normalizedLayerType) {
    return { sharedKVSourceLayerIdx: null, storeSharedKV: false };
  }

  let sourceLayerIdx = null;
  for (let index = firstKvSharedLayerIdx - 1; index >= 0; index -= 1) {
    if (normalizeLayerType(layerTypes[index]) === normalizedLayerType) {
      sourceLayerIdx = index;
      break;
    }
  }
  if (sourceLayerIdx == null) {
    return { sharedKVSourceLayerIdx: null, storeSharedKV: false };
  }

  if (layerIdx >= firstKvSharedLayerIdx) {
    return { sharedKVSourceLayerIdx: sourceLayerIdx, storeSharedKV: false };
  }

  return {
    sharedKVSourceLayerIdx: null,
    storeSharedKV: layerIdx === sourceLayerIdx,
  };
}

export function hasPerLayerInputBlock(config) {
  const hiddenSizePerLayerInput = Number(config?.hiddenSizePerLayerInput ?? 0);
  return Number.isFinite(hiddenSizePerLayerInput) && hiddenSizePerLayerInput > 0;
}

export function resolveLayerScalarValue(layerScalar) {
  if (layerScalar == null) {
    return 1;
  }
  if (!(layerScalar instanceof Float32Array) || layerScalar.length === 0) {
    throw new Error(
      'Gemma 4 per-layer input layer_scalar must be CPU-resident Float32Array data. ' +
      'Re-convert or reload the model with the updated loader.'
    );
  }
  const value = Number(layerScalar[0]);
  if (!Number.isFinite(value)) {
    throw new Error(`Gemma 4 layer_scalar must be finite; got "${String(layerScalar[0])}".`);
  }
  return value;
}

export async function applyLayerScalar(layerIdx, tensor, size, context, layerWeights) {
  const layerScalar = resolveLayerScalarValue(layerWeights?.layerScalar ?? null);
  if (layerScalar === 1) {
    return tensor;
  }
  return context.recorder
    ? recordScale(context.recorder, tensor, layerScalar, { count: size })
    : runScale(tensor, layerScalar, { count: size });
}

async function debugLayerTensor(context, layerIdx, label, tensor, numTokens, hiddenSize) {
  if (!context.debugCheckBuffer) return;
  if (!shouldDebugLayerOutput(layerIdx, context.debugLayers)) return;
  if (!isGpuBufferInstance(tensor?.buffer)) return;
  await context.debugCheckBuffer(tensor.buffer, `L${layerIdx} ${label} (GPU)`, numTokens, hiddenSize);
}

export async function applyPerLayerInputBlock(layerIdx, hiddenTensor, numTokens, size, context, layerWeights) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  if (!hasPerLayerInputBlock(config)) {
    return hiddenTensor;
  }

  const hiddenSizePerLayerInput = Number(config.hiddenSizePerLayerInput);
  const perLayerInputBuffer = context.perLayerInputBuffer ?? null;
  if (!perLayerInputBuffer) {
    throw new Error(
      `Gemma 4 layer ${layerIdx} requires a per-layer input buffer, but context.perLayerInputBuffer was not set.`
    );
  }
  if (!layerWeights?.perLayerInputGate || !layerWeights?.perLayerProjection || !layerWeights?.postPerLayerInputNorm) {
    throw new Error(
      `Gemma 4 layer ${layerIdx} is missing per-layer input weights. ` +
      'Expected per_layer_input_gate.weight, per_layer_projection.weight, and post_per_layer_input_norm.weight.'
    );
  }

  const residualTensor = hiddenTensor;
  let gateTensor = null;
  let activatedTensor = null;
  let projectedTensor = null;
  let normalizedTensor = null;
  let outputTensor = null;

  try {
    gateTensor = await processLayerPerLayerInputGate(
      layerIdx,
      hiddenTensor,
      numTokens,
      hiddenSizePerLayerInput,
      context,
      layerWeights
    );
    await runProbes('per_layer_input_gate', gateTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: hiddenSizePerLayerInput,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: gateTensor.dtype,
    });
    await debugLayerTensor(context, layerIdx, 'per-layer input gate', gateTensor, numTokens, hiddenSizePerLayerInput);

    const perLayerInputTensor = createPerLayerInputTensor(
      perLayerInputBuffer,
      numTokens,
      hiddenSizePerLayerInput,
      hiddenTensor.dtype
    );
    activatedTensor = await doGeLU(perLayerInputTensor, {
      size: numTokens * hiddenSizePerLayerInput,
      gate: gateTensor,
      label: `L${layerIdx}.per_layer_input_activation`,
      layerIdx,
    }, recorder);
    await runProbes('per_layer_input_activation', activatedTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: hiddenSizePerLayerInput,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: activatedTensor.dtype,
    });
    await debugLayerTensor(context, layerIdx, 'per-layer input activation', activatedTensor, numTokens, hiddenSizePerLayerInput);
    releaseOrTrack(recorder, gateTensor.buffer, decodeBuffers);
    gateTensor = null;

    projectedTensor = await processLayerPerLayerInputProjection(
      layerIdx,
      activatedTensor,
      numTokens,
      context,
      layerWeights
    );
    await runProbes('per_layer_input_projection', projectedTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: config.hiddenSize,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: projectedTensor.dtype,
    });
    await debugLayerTensor(context, layerIdx, 'per-layer input projection', projectedTensor, numTokens, config.hiddenSize);
    releaseOrTrack(recorder, activatedTensor.buffer, decodeBuffers);
    activatedTensor = null;

    const postNormWeight = getNormWeightBuffer(
      layerWeights.postPerLayerInputNorm,
      `L${layerIdx}.post_per_layer_input_norm`,
      weightConfig,
      debugFlags
    );
    normalizedTensor = await doRMSNorm(projectedTensor, postNormWeight, config.rmsNormEps, {
      batchSize: numTokens,
      hiddenSize: config.hiddenSize,
      label: `L${layerIdx}.post_per_layer_input_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);
    await runProbes('post_per_layer_input_norm', normalizedTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: config.hiddenSize,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: normalizedTensor.dtype,
    });
    await debugLayerTensor(context, layerIdx, 'post per-layer input norm', normalizedTensor, numTokens, config.hiddenSize);
    if (!isGpuBufferInstance(layerWeights.postPerLayerInputNorm)) {
      releaseOrTrack(recorder, postNormWeight, decodeBuffers);
    }
    releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
    projectedTensor = null;

    outputTensor = await doResidualAdd(normalizedTensor, residualTensor, size, recorder, {
      label: `L${layerIdx}.per_layer_input_residual`,
      layerIdx,
      executionPolicies: context.executionPolicies ?? null,
    });
    releaseOrTrack(recorder, normalizedTensor.buffer, decodeBuffers);
    normalizedTensor = null;

    await runProbes('post_per_layer_input', outputTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: config.hiddenSize,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: outputTensor.dtype,
    });
    await debugLayerTensor(context, layerIdx, 'post per-layer input', outputTensor, numTokens, config.hiddenSize);

    return outputTensor;
  } catch (error) {
    if (outputTensor?.buffer) releaseOrTrack(recorder, outputTensor.buffer, decodeBuffers);
    if (normalizedTensor?.buffer) releaseOrTrack(recorder, normalizedTensor.buffer, decodeBuffers);
    if (projectedTensor?.buffer) releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
    if (activatedTensor?.buffer) releaseOrTrack(recorder, activatedTensor.buffer, decodeBuffers);
    if (gateTensor?.buffer) releaseOrTrack(recorder, gateTensor.buffer, decodeBuffers);
    throw error;
  }
}

async function processLayerPerLayerInputGate(
  layerIdx,
  hiddenTensor,
  numTokens,
  hiddenSizePerLayerInput,
  context,
  layerWeights
) {
  return doMatmul(
    hiddenTensor,
    getWeightBuffer(layerWeights.perLayerInputGate, `L${layerIdx}.per_layer_input_gate`),
    numTokens,
    hiddenSizePerLayerInput,
    context.config.hiddenSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.per_layer_input_gate`,
      layerIdx,
      kernelPath: context.kernelPath ?? null,
      role: 'per_layer_input_gate',
      outputDtype: hiddenTensor.dtype,
    },
    context.recorder
  );
}

async function processLayerPerLayerInputProjection(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights
) {
  const projectionWeight = resolveDensePleProjectionWeight(
    getWeightBuffer(layerWeights.perLayerProjection, `L${layerIdx}.per_layer_projection`),
    `L${layerIdx}.per_layer_projection`
  );
  return doMatmul(
    inputTensor,
    projectionWeight,
    numTokens,
    context.config.hiddenSize,
    context.config.hiddenSizePerLayerInput,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.per_layer_projection`,
      layerIdx,
      kernelPath: context.kernelPath ?? null,
      role: 'per_layer_projection',
      outputDtype: inputTensor.dtype,
    },
    context.recorder
  );
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

  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  assertSupportedLayerRuntime(layerIdx, config);
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  // Determine activation dtype from context (defaults to f32)

  const activationDtype = resolveActivationDtype(context.activationDtype);

  // Wrap input buffer as Tensor for dtype-aware processing
  const inputTensor = createTensor(inputBuffer, activationDtype, [numTokens, hiddenSize], 'layer_input');

  const layerWeights = (weights.get(`layer_${layerIdx}`));
  const sandwichNorm = detectSandwichNorm(config);
  const lastTokenIdx = Math.max(0, numTokens - 1);

  if (context.pipelinePlan) {
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
      debugProbes: context.debugProbes,
      operatorDiagnostics: context.operatorDiagnostics,
      recorder: recorder ?? null,
    });
  } else {
    let attentionNumHeads = numHeads;
    let attentionHeadDim = resolveAttentionHeadDim(config, layerType);
    let attentionNumKVHeads = resolveAttentionNumKVHeads(config, layerType, layerWeights, attentionHeadDim);
    let disableRoPE = false;
    let queryKeyNorm = config.queryKeyNorm;
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
      residualTensor: (numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm))
        ? inputTensor
        : null,
      attnSoftcap: config.attnLogitSoftcapping === null ? 0 : config.attnLogitSoftcapping,
      queryPreAttnScalar: config.queryPreAttnScalar,
      queryKeyNorm,
      valueNorm: config.valueNorm,
      attentionOutputGate: config.attentionOutputGate,
      causalAttention: config.causalAttention,
      multimodalBidirectionalSpan: isSlidingLayerType(layerType)
        ? (context.multimodalBidirectionalSpan ?? null)
        : null,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
      ropeRotaryDim: resolveAttentionRotaryDim(config, layerType),
      ropeFrequencyBaseDim: resolveAttentionFrequencyBaseDim(config, layerType),
      ropeInterleaved: config.ropeInterleaved,
      tokenIds: context.currentTokenIds ?? null,
      kernelPath: context.kernelPath ?? null,
      sessionSettings: config.sessionSettings ?? null,
      disableRoPE,
      sharedKVSourceLayerIdx,
      storeSharedKV,
    };

    validateAttnConfig(attnConfig, `L${layerIdx}`);

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
      executionPolicies: context.executionPolicies ?? null,
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

  // 2. Handle residual connection based on architecture

  if (residualFused) {
    postAttn = attnOutput;
    if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
      const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
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
    const normalizedAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);
    postAttn = await doResidualAdd(normalizedAttn, inputTensor, size, recorder, {
      label: `L${layerIdx}.post_attn_residual`,
      layerIdx,
      executionPolicies: context.executionPolicies ?? null,
    });

    if (!isGpuBufferInstance(layerWeights.postAttentionNorm) && !isWeightBuffer(layerWeights.postAttentionNorm)) releaseOrTrack(recorder, normWeightBuf);
    releaseOrTrack(recorder, normalizedAttn.buffer, context.decodeBuffers);
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
  if (sandwichNorm.useSandwichNorm) {
    outputTensor = await processFFNWithSandwichNorm(layerIdx, postAttn, numTokens, size, context, layerWeights, sandwichNorm);
  } else {
    outputTensor = await processFFNStandard(layerIdx, postAttn, numTokens, size, context, layerWeights, fusedResidualForFFN);
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

  // Keep activation dtype consistent across layers. Some FFN paths can emit f32
  // tensors even when the execution plan is f16; leaving that unnormalized causes
  // downstream kernels to decode the buffer with the wrong dtype contract.
  let finalOutput = outputTensor;
  if (outputTensor.dtype !== activationDtype) {
    finalOutput = await doCast(
      outputTensor,
      activationDtype,
      recorder
    );
    releaseOrTrack(recorder, outputTensor.buffer, context.decodeBuffers);
  }
  const scaledOutput = await applyLayerScalar(layerIdx, finalOutput, size, context, layerWeights);
  if (scaledOutput.buffer !== finalOutput.buffer) {
    releaseOrTrack(recorder, finalOutput.buffer, context.decodeBuffers);
    finalOutput = scaledOutput;
  }
  await debugLayerTensor(context, layerIdx, 'final layer output', finalOutput, numTokens, hiddenSize);

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
