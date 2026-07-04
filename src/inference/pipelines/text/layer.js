

import { log, trace } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { getDevice } from '../../../gpu/device.js';
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
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { recordCheckFiniteness } from '../../../gpu/kernels/check-finiteness.js';
import { RMSNORM_PAIR_CACHE_LIMIT } from '../../../gpu/kernel-selector.js';
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

function shouldUseSandwichRMSNormPairFusion({
  context,
  sandwichNorm,
  layerWeights,
  numTokens,
  hiddenSize,
  attnOutput,
  inputTensor,
}) {
  const mergedSession = getRuntimeConfig()?.inference?.session;
  if (mergedSession?.useSandwichRMSNormPairFusion !== true) {
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

function shouldUsePostFfnNextInputRMSNormPairFusion({
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
}) {
  const mergedSession = getRuntimeConfig()?.inference?.session;
  if (mergedSession?.usePostFfnNextInputRMSNormPairFusion !== true) {
    return false;
  }
  if (numTokens !== 1 || context.phase !== 'decode' || context.diffusionGemmaDecoder === true) {
    return false;
  }
  if (context.debug === true || context.debugProbes?.length || context.operatorDiagnostics?.enabled === true) {
    return false;
  }
  if (!context.decodeBuffers || context.pipelinePlan || hasPerLayerInputBlock(config)) {
    return false;
  }
  if (layerScalar !== 1 || activationDtype !== 'f32') {
    return false;
  }
  const nextLayerIdx = layerIdx + 1;
  if (nextLayerIdx >= config.numLayers) {
    return false;
  }
  const nextLayerType = config.layerTypes?.[nextLayerIdx];
  if (isConvLayerType(nextLayerType) || isLinearLayerType(nextLayerType)) {
    return false;
  }
  if (!sandwichNorm.useSandwichNorm || !sandwichNorm.hasPostFeedforwardNorm) {
    return false;
  }
  if (!layerWeights?.postFeedforwardNorm || !nextLayerWeights?.inputNorm) {
    return false;
  }
  if (config.useMoE && isMoELayer(layerIdx, config)) {
    return false;
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0 || hiddenSize > RMSNORM_PAIR_CACHE_LIMIT) {
    throw new Error(
      `usePostFfnNextInputRMSNormPairFusion requires hiddenSize in 1..${RMSNORM_PAIR_CACHE_LIMIT}; got ${String(hiddenSize)}.`
    );
  }
  return true;
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
    // The gate weight may be stored as f32 (small projection, not always quantized),
    // forcing the matmul into the f32 variant whose output dtype is fixed by the
    // kernel registry, not the caller's requestedOutputDtype. doGeLU below dispatches
    // gelu_f16 when input is f16 and reads `gate: array<f16>` — binding an f32 buffer
    // there reinterprets bytes as 2x f16 per f32 element and produces NaN/garbage.
    if (gateTensor.dtype !== hiddenTensor.dtype) {
      const widened = gateTensor;
      gateTensor = await doCast(widened, hiddenTensor.dtype, recorder);
      releaseOrTrack(recorder, widened.buffer, decodeBuffers);
    }
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

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('No GPU device available');

  assertSupportedLayerRuntime(layerIdx, config);
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;
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
      debugProbes: context.debugProbes,
      operatorDiagnostics: context.operatorDiagnostics,
      recorder: recorder ?? null,
    });
  } else {
    let attentionNumHeads = numHeads;
    let attentionHeadDim = resolveAttentionHeadDim(config, layerType);
    let attentionNumKVHeads = resolveAttentionNumKVHeads(config, layerType, layerWeights, attentionHeadDim);
    let disableRoPE = false;
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
      queryKeyNorm,
      queryKeyNormWeightLayers: config.queryKeyNormWeightLayers,
      valueNorm: config.valueNorm,
      attentionOutputGate: config.attentionOutputGate,
      outputGateType: config.outputGateType ?? null,
      causalAttention: diffusionGemmaDecoder ? false : config.causalAttention,
      multimodalBidirectionalSpan: !diffusionGemmaDecoder && isSlidingLayerType(layerType)
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
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        residual: inputTensor,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
    } else {
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
	      residualBranchScale
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
