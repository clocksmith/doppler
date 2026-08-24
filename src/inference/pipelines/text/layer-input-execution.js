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
import { selectRuleValue } from '../../../rules/rule-registry.js';
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

export function shouldUsePostFfnNextInputRMSNormPairFusion({
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
  if (config.normalizationType === 'layernorm' || !postNormContractMatchesBase(config)) return false;
  if (context.usePostFfnNextInputRMSNormPairFusion !== true) {
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
  if (isConvLayerType(nextLayerType)) {
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

export function shouldUseStandardPostFfnNextInputRMSNormPairFusion({
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
}) {
  if (config.normalizationType === 'layernorm') return false;
  if (context.usePostFfnNextInputRMSNormPairFusion !== true) {
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
  if (sandwichNorm.useSandwichNorm || layerScalar !== 1 || residualBranchScale !== 1 || activationDtype !== 'f32') {
    return false;
  }
  if (getKernelCapabilities()?.hasSubgroups !== true) {
    return false;
  }
  const nextLayerIdx = layerIdx + 1;
  if (nextLayerIdx >= config.numLayers) {
    return false;
  }
  const nextLayerType = config.layerTypes?.[nextLayerIdx];
  if (isConvLayerType(nextLayerType)) {
    return false;
  }
  if (!nextLayerWeights?.inputNorm) {
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

export function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

export function isConvLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'conv'
    || normalized === 'convolution'
    || normalized === 'liv_conv'
    || normalized === 'liv_convolution';
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

export async function debugLayerTensor(context, layerIdx, label, tensor, numTokens, hiddenSize) {
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
  let residualAddTensor = null;
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

    residualAddTensor = normalizedTensor;
    if (residualAddTensor.dtype !== residualTensor.dtype) {
      residualAddTensor = await doCast(residualAddTensor, residualTensor.dtype, recorder);
    }
    outputTensor = await doResidualAdd(residualAddTensor, residualTensor, size, recorder, {
      label: `L${layerIdx}.per_layer_input_residual`,
      layerIdx,
      executionPolicies: context.executionPolicies ?? null,
    });
    if (residualAddTensor.buffer !== normalizedTensor.buffer) {
      releaseOrTrack(recorder, residualAddTensor.buffer, decodeBuffers);
    }
    residualAddTensor = null;
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
    if (residualAddTensor?.buffer && residualAddTensor.buffer !== normalizedTensor?.buffer) {
      releaseOrTrack(recorder, residualAddTensor.buffer, decodeBuffers);
    }
    if (normalizedTensor?.buffer) releaseOrTrack(recorder, normalizedTensor.buffer, decodeBuffers);
    if (projectedTensor?.buffer) releaseOrTrack(recorder, projectedTensor.buffer, decodeBuffers);
    if (activatedTensor?.buffer) releaseOrTrack(recorder, activatedTensor.buffer, decodeBuffers);
    if (gateTensor?.buffer) releaseOrTrack(recorder, gateTensor.buffer, decodeBuffers);
    throw error;
  }
}

export async function processLayerPerLayerInputGate(
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

export async function processLayerPerLayerInputProjection(
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
