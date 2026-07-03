import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, BufferUsage, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { castF16ToF32, castF32ToF16 } from '../../../gpu/kernels/cast.js';
import {
  runMatmul,
  runSiLU,
  runSiLURowSplit,
  runGeLU,
  dequantizeMXFP4Expert,
  runBiasAdd,
  runRMSNorm,
  runScale,
  runSoftmaxTopK,
  runMoEGather,
  runMoEBuildTokenOffsets,
  runScatterAddDynamic,
  runSwiGLURowsplitBias,
  runGemma4RouteQ4MatmulF16A,
  runScatterAddRoutesF16ExpertScale,
} from '../../../gpu/kernel-selector.js';
import { getBuffer, getWeightDtype, isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { trace, isTraceEnabled } from '../../../debug/index.js';
import { f16ToF32Array } from '../../kv-cache/types.js';
import { resolveMaxTokensPerExpert, getCachedDequant, setCachedDequant, getDequantCacheStats } from './moe-cache.js';
import { ensureExpertLoaded } from './moe-helpers.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import {
  validateMoeShape,
  resolveMoeExecutionProfile,
  resolveMoeIntermediateSize,
  resolveMoeVendorProfile,
  resolveMoeKernelPathProfile,
} from './moe-shape-validator.js';
import { assertImplicitDtypeTransitionAllowed } from './dtype-contract.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../config/schema/index.js';

function resolveMoEActiveExpertSelection() {
  const selection = getRuntimeConfig()?.inference?.moe?.routing?.activeExpertSelection;
  if (selection === 'all' || selection === 'topk-readback' || selection === 'topk-route') {
    return selection;
  }
  throw new Error(
    '[MoE] runtime.inference.moe.routing.activeExpertSelection must be ' +
    `"all", "topk-readback", or "topk-route", got ${String(selection)}.`
  );
}

async function resolveActiveExpertSchedule(indicesBuffer, numTokens, numExperts, topK, maxTokensPerExpert) {
  const selection = resolveMoEActiveExpertSelection();
  if (selection === 'all') {
    return {
      selection,
      activeExperts: Array.from({ length: numExperts }, (_, expertIdx) => expertIdx),
      tokenCounts: null,
    };
  }

  const indicesBytes = numTokens * topK * 4;
  const indicesData = await readBuffer(indicesBuffer, indicesBytes);
  return buildActiveExpertScheduleFromIndices(
    new Uint32Array(indicesData),
    numExperts,
    maxTokensPerExpert,
    selection
  );
}

export function buildActiveExpertScheduleFromIndices(
  indices,
  numExperts,
  maxTokensPerExpert,
  selection = 'topk-readback'
) {
  const tokenCounts = new Uint32Array(numExperts);

  for (let i = 0; i < indices.length; i++) {
    const expertIdx = indices[i];
    if (expertIdx >= numExperts) {
      throw new Error(
        `[MoE] Top-K routing produced expert index ${expertIdx} outside numExperts=${numExperts}.`
      );
    }
    tokenCounts[expertIdx] += 1;
  }

  const activeExperts = [];
  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCounts[expertIdx];
    if (count === 0) {
      continue;
    }
    if (count > maxTokensPerExpert) {
      throw new Error(
        `[MoE] Expert ${expertIdx} received ${count} tokens but maxTokensPerExpert=${maxTokensPerExpert}. ` +
        'Increase runtime.inference.moe.routing.maxTokensPerExpert or its headroom/cap settings.'
      );
    }
    activeExperts.push(expertIdx);
  }

  return { selection, activeExperts, tokenCounts };
}

function resolvePerExpertScaleBuffer(device, value) {
  if (value == null) {
    return { buffer: null, ownedBuffer: null };
  }
  if (value instanceof Float32Array) {
    const buffer = acquireBuffer(value.byteLength, BufferUsage.STORAGE_READ, 'moe_per_expert_scale_f32');
    try {
      device.queue.writeBuffer(buffer, 0, value);
    } catch (error) {
      releaseBuffer(buffer);
      throw error;
    }
    return { buffer, ownedBuffer: buffer };
  }

  const dtype = getWeightDtype(value);
  if (dtype != null && dtype !== 'f32') {
    throw new Error(`[MoE] per-expert router scale must be f32 for scatter-add, got ${dtype}.`);
  }
  const buffer = getBuffer(value);
  if (!isGpuBufferInstance(buffer)) {
    throw new Error('[MoE] per-expert router scale must resolve to a GPUBuffer.');
  }
  return { buffer, ownedBuffer: null };
}

const MOE_ROUTE_EXECUTORS = Object.freeze({
  'gemma4-route': runGemma4RouteExperts,
});

const MOE_EXPERT_EXECUTORS = Object.freeze({
  'gpt-oss': runGptOssProfileExpert,
  'gemma4-packed': runGemma4ProfileExpert,
  mixtral: runMixtralProfileExpert,
});

function requireMoeExecutor(registry, id, label) {
  const executor = registry[id];
  if (typeof executor !== 'function') {
    throw new Error(`[MoE] Unknown ${label} "${String(id)}".`);
  }
  return executor;
}

function assertMoeExpertWeights(moeProfile, weights, expertKey) {
  if (moeProfile.expertExecutor === 'gemma4-packed' && !weights.gateUp) {
    throw new Error(`[MoE] Missing Gemma-style packed weights for ${expertKey}`);
  }
  if (moeProfile.expertExecutor === 'gemma4-packed' && !weights.down) {
    throw new Error(`[MoE] Missing Gemma-style packed weights for ${expertKey}`);
  }
  if (moeProfile.expertExecutor === 'mixtral' && (!weights.gate || !weights.up || !weights.down)) {
    throw new Error(`[MoE] Missing Mixtral weights for ${expertKey}`);
  }
}

async function runGptOssProfileExpert(args) {
  return runGptOssExpert(
    args.gathered,
    args.expertOutputs,
    args.weights,
    args.layerIdx,
    args.expertIdx,
    args.count,
    args.inputOffset,
    args.outputOffset,
    args.hiddenSize,
    args.intermediateSize,
    args.numExperts,
    args.activationDtype,
    args.swigluLimit,
    args.kernelPath,
    args.executionPolicies,
    args.modelType,
    args.vendorProfile,
    args.moeKernelPathProfile
  );
}

async function runGemma4ProfileExpert(args) {
  return runGemma4Expert(
    args.gathered,
    args.expertOutputs,
    args.weights,
    args.count,
    args.inputOffset,
    args.outputOffset,
    args.hiddenSize,
    args.intermediateSize,
    args.activationDtype,
    args.swigluLimit,
    args.kernelPath
  );
}

async function runMixtralProfileExpert(args) {
  return runMixtralExpert(
    args.gathered,
    args.expertOutputs,
    args.weights,
    args.count,
    args.inputOffset,
    args.outputOffset,
    args.hiddenSize,
    args.intermediateSize,
    args.hiddenActivation,
    args.activationDtype,
    args.swigluLimit,
    args.kernelPath
  );
}

export async function moeFeedForwardGPU(
  inputBuffer,
  numTokens,
  config,
  moeRouter,
  expertWeights,
  expertLoader,
  layerIdx,
  layerRouterWeights
) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device for MoE');

  const { hiddenSize, numExperts, moeTopK, hiddenActivation } = config;
  const expertFormat = config.expertFormat;
  const swigluLimit = config.swigluLimit;
  const kernelPath = config.kernelPath ?? null;
  if (!expertFormat) {
    throw new Error('MoE expertFormat is required in config.');
  }
  if (swigluLimit === undefined) {
    throw new Error('MoE swigluLimit must be explicitly set (null or number).');
  }
  const topK = moeTopK ?? moeRouter.topK;
  if (topK == null) {
    throw new Error('MoE topK is required in config.');
  }
  if (config.modelType == null) {
    throw new Error('MoE config.modelType is required; got null/undefined.');
  }
  const modelType = config.modelType;
  const moeProfile = resolveMoeExecutionProfile(config, { modelType });
  const intermediateSize = resolveMoeIntermediateSize(config, moeProfile);
  validateMoeShape(
    { hiddenSize, intermediateSize, moeTopK: topK, numExperts, expertFormat },
    { modelType, moeProfile }
  );
  const vendorProfile = resolveMoeVendorProfile(moeProfile);
  const caps = getKernelCapabilities();
  if (moeProfile.requiresShaderF16 && !caps.hasF16) {
    throw new Error(
      `[MoE] ${moeProfile.label} requires shader-f16 support. ` +
      `Adapter: ${caps.adapterInfo?.vendor ?? 'unknown'} ${caps.adapterInfo?.architecture ?? ''}`.trim()
    );
  }
  const activationDtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', {
    dtype: config.activationDtype,
  });

  if (!moeRouter || !moeRouter.gateWeight) {
    throw new Error('MoE router not initialized');
  }

  const perfEnabled = isTraceEnabled('perf');
  const perfMark = () => (perfEnabled ? performance.now() : 0);
  const perfLog = (label, start, data) => {
    if (!perfEnabled) return;
    trace.perf(`${label}: ${(performance.now() - start).toFixed(2)}ms`, data);
  };

  const inputTensor = createTensor(inputBuffer, activationDtype, [numTokens, hiddenSize], 'moe_input');
  const routerSourceTensor = createTensor(
    config.routerInputBuffer ?? inputBuffer,
    config.routerInputDtype ?? activationDtype,
    [numTokens, hiddenSize],
    'moe_router_input'
  );
  let logitsBuffer = null;
  let indicesBuffer = null;
  let weightsBuffer = null;
  let gathered = null;
  let tokenCounts = null;
  let tokenMap = null;
  let expertOutputs = null;
  let tokenOffsets = null;
  let outputTensor = null;
  let routerNormTensor = null;
  let routerScaledTensor = null;
  let ownedPerExpertScaleBuffer = null;
  let activeExpertSchedule = null;

  const layerRouter = layerRouterWeights?.get(layerIdx) || null;
  if (layerRouter) {
    moeRouter.loadWeights(
      layerRouter.weight,
      layerRouter.bias || null,
      layerRouter.scale || null,
      layerRouter.perExpertScale || null
    );
  }

  try {
    const needsRouterScale = moeProfile.routerScaleMode === 'required'
      || layerRouter?.scale != null
      || layerRouter?.perExpertScale != null;
    let routerInputTensor = routerSourceTensor;
    if (needsRouterScale) {
      if (!layerRouter?.scale) {
        throw new Error(`[MoE] ${moeProfile.label} router scale missing for layer ${layerIdx}.`);
      }
      if (!layerRouter?.perExpertScale) {
        throw new Error(`[MoE] ${moeProfile.label} per-expert router scale missing for layer ${layerIdx}.`);
      }
      if (!Number.isFinite(config.rmsNormEps) || config.rmsNormEps <= 0) {
        throw new Error(`[MoE] ${moeProfile.label} router RMSNorm eps is invalid: ${String(config.rmsNormEps)}.`);
      }
      routerNormTensor = await runRMSNorm(
        inputTensor,
        layerRouter.scale,
        config.rmsNormEps,
        {
          batchSize: numTokens,
          hiddenSize,
          rmsNormWeightOffset: false,
        }
      );
      routerScaledTensor = await runScale(
        routerNormTensor,
        1 / Math.sqrt(hiddenSize),
        { count: numTokens * hiddenSize }
      );
      releaseBuffer(routerNormTensor.buffer);
      routerNormTensor = null;
      routerInputTensor = routerScaledTensor;
    }

    let stepStart = perfMark();
    logitsBuffer = await moeRouter.computeRouterLogitsGPU(routerInputTensor.buffer, numTokens, null, {
      inputDtype: routerInputTensor.dtype,
      outputDtype: activationDtype,
    });
    if (routerScaledTensor) {
      releaseBuffer(routerScaledTensor.buffer);
      routerScaledTensor = null;
    }
  const logitsDtype = moeRouter.lastLogitsDtype ?? activationDtype;
  perfLog(`MoE L${layerIdx} router`, stepStart, { numTokens, logitsDtype });

  if (isTraceEnabled('buffers')) {
    const logitsBytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
    const logitsBytes = numTokens * numExperts * logitsBytesPerElement;
    const logitsData = await readBuffer(logitsBuffer, logitsBytes);
    let logits;
    if (logitsDtype === 'f16') {
      logits = f16ToF32Array(new Uint16Array(logitsData));
    } else {
      logits = new Float32Array(logitsData);
    }
    let min = Infinity;
    let max = -Infinity;
    let nanCount = 0;
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (!Number.isFinite(v)) {
        nanCount += 1;
        continue;
      }
      if (v < min) min = v;
      if (v > max) max = v;
    }
    trace.buffers(`MoE L${layerIdx} router_logits`, { min, max, nanCount, dtype: logitsDtype });
  }

  const moeKernelPathProfile = await resolveMoeKernelPathProfile(moeProfile, {
    hasF16: caps.hasF16,
    hasSubgroups: caps.hasSubgroups,
    routerDtype: logitsDtype,
    inputDtype: logitsDtype,
    weightsDtype: activationDtype,
    outputDtype: activationDtype,
    groupSize: 32,
    tileShape: vendorProfile.dequantTileShape,
  });

  stepStart = perfMark();
    ({ indices: indicesBuffer, weights: weightsBuffer } = await runSoftmaxTopK(
      logitsBuffer,
      numTokens,
      numExperts,
      topK,
      {
        normalize: moeRouter.normalizeWeights,
        inputDtype: logitsDtype,
        weightsDtype: activationDtype,
        modelType,
      }
    ));
  perfLog(`MoE L${layerIdx} topk`, stepStart, {
    topK,
    modelType,
    routerTopKKernel: moeKernelPathProfile?.routerTopK ?? null,
  });

  if (isTraceEnabled('buffers')) {
    const indicesData = await readBuffer(indicesBuffer, numTokens * topK * 4);
    const indices = new Uint32Array(indicesData);
    let minIdx = Number.MAX_SAFE_INTEGER;
    let maxIdx = 0;
    let outOfRange = 0;
    for (let i = 0; i < indices.length; i++) {
      const v = indices[i];
      if (v < minIdx) minIdx = v;
      if (v > maxIdx) maxIdx = v;
      if (v >= numExperts) outOfRange += 1;
    }
    trace.buffers(`MoE L${layerIdx} topk_indices`, {
      minIdx,
      maxIdx,
      outOfRange,
      numExperts,
    });

    const weightsBytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
    const weightsBytes = numTokens * topK * weightsBytesPerElement;
    const weightsData = await readBuffer(weightsBuffer, weightsBytes);
    let weights;
    if (activationDtype === 'f16') {
      weights = f16ToF32Array(new Uint16Array(weightsData));
    } else {
      weights = new Float32Array(weightsData);
    }
    let minW = Infinity;
    let maxW = -Infinity;
    let nanW = 0;
    for (let i = 0; i < weights.length; i++) {
      const v = weights[i];
      if (!Number.isFinite(v)) {
        nanW += 1;
        continue;
      }
      if (v < minW) minW = v;
      if (v > maxW) maxW = v;
    }
    trace.buffers(`MoE L${layerIdx} topk_weights`, { minW, maxW, nanW, dtype: activationDtype });
  }

    releaseBuffer(logitsBuffer);
    logitsBuffer = null;

  const activeExpertSelection = resolveMoEActiveExpertSelection();
  if (activeExpertSelection === 'topk-route') {
    if (moeProfile.topkRouteExecutor == null) {
      throw new Error(`[MoE] topk-route active expert selection is not supported by profile "${moeProfile.id}".`);
    }
    const routeExecutor = requireMoeExecutor(MOE_ROUTE_EXECUTORS, moeProfile.topkRouteExecutor, 'route executor');
    stepStart = perfMark();
    await ensureExpertLoaded(layerIdx, 0, expertWeights, expertLoader);
    const routeWeights = expertWeights.get(`layer_${layerIdx}_expert_0`);
    perfLog(`MoE L${layerIdx} route_weight_load`, stepStart, { expertFormat, topK });
    stepStart = perfMark();
    outputTensor = await routeExecutor({
      inputTensor,
      indicesBuffer,
      weightsBuffer,
      layerRouter,
      weights: routeWeights,
      expectedExpertFormat: expertFormat,
      profile: moeProfile,
      layerIdx,
      numTokens,
      topK,
      hiddenSize,
      intermediateSize,
      activationDtype,
      swigluLimit,
    });
    perfLog(`MoE L${layerIdx} route_experts`, stepStart, {
      numTokens,
      topK,
      numRoutes: numTokens * topK,
      hiddenSize,
      intermediateSize,
    });
    return outputTensor.buffer;
  }

  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
  const bytesPerToken = hiddenSize * bytesPerElement;
  let maxTokensPerExpert = resolveMaxTokensPerExpert(numTokens, numExperts, topK, hiddenSize, activationDtype);
  if (vendorProfile.maxTokensPerExpertScale !== 1.0) {
    maxTokensPerExpert = Math.max(
      1,
      Math.round(maxTokensPerExpert * vendorProfile.maxTokensPerExpertScale)
    );
  }

    stepStart = perfMark();
    activeExpertSchedule = await resolveActiveExpertSchedule(
      indicesBuffer,
      numTokens,
      numExperts,
      topK,
      maxTokensPerExpert
    );
  perfLog(`MoE L${layerIdx} active_experts`, stepStart, {
    selection: activeExpertSchedule.selection,
    activeExperts: activeExpertSchedule.activeExperts.length,
    numExperts,
    maxTokensPerExpert,
  });

    stepStart = perfMark();
    ({ gathered, tokenCounts, tokenMap } = await runMoEGather(
      inputTensor,
      indicesBuffer,
      numTokens,
      hiddenSize,
      numExperts,
      topK,
      { maxTokensPerExpert }
    ));
  perfLog(`MoE L${layerIdx} gather`, stepStart, { maxTokensPerExpert });

    expertOutputs = acquireBuffer(
      numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement,
      undefined,
      'moe_expert_outputs_gathered'
    );

    stepStart = perfMark();
    tokenOffsets = await runMoEBuildTokenOffsets(
      tokenCounts,
      tokenMap,
      numTokens,
      numExperts,
      topK,
      maxTokensPerExpert
    );
  perfLog(`MoE L${layerIdx} offsets_kernel`, stepStart, {
    totalSlots: numExperts * maxTokensPerExpert,
    routingSlots: numTokens * topK,
  });

    releaseBuffer(tokenCounts);
    tokenCounts = null;

  const expertStrideBytes = maxTokensPerExpert * bytesPerToken;
  const rowsPerExpert = maxTokensPerExpert;

  const scheduledExperts = activeExpertSchedule?.activeExperts ?? [];
  const scheduledTokenCounts = activeExpertSchedule?.tokenCounts ?? null;
  for (const expertIdx of scheduledExperts) {
    const count = scheduledTokenCounts ? scheduledTokenCounts[expertIdx] : rowsPerExpert;
    if (count <= 0) {
      continue;
    }

    stepStart = perfMark();
    await ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader);
    perfLog(`MoE L${layerIdx} expert_load`, stepStart, { expertIdx, count });
    const expertKey = `layer_${layerIdx}_expert_${expertIdx}`;
    const weights = expertWeights.get(expertKey);
    if (!weights) {
      throw new Error(`[MoE] Missing expert weights for ${expertKey}`);
    }
    if (!weights.expertFormat) {
      throw new Error(`[MoE] Expert ${expertKey} missing expertFormat.`);
    }
    assertMoeExpertWeights(moeProfile, weights, expertKey);

    const inputOffset = expertIdx * expertStrideBytes;
    const outputOffset = expertIdx * expertStrideBytes;

    stepStart = perfMark();
    if (weights.expertFormat !== expertFormat) {
      throw new Error(
        `[MoE] Expert format mismatch for ${expertKey}: ` +
        `weights=${weights.expertFormat}, config=${expertFormat}`
      );
    }

    const expertExecutor = requireMoeExecutor(MOE_EXPERT_EXECUTORS, moeProfile.expertExecutor, 'expert executor');
    await expertExecutor({
        gathered,
        expertOutputs,
        weights,
        layerIdx,
        expertIdx,
        count,
        inputOffset,
        outputOffset,
        hiddenSize,
        intermediateSize,
        numExperts,
        activationDtype,
        swigluLimit,
        kernelPath,
        executionPolicies: config.executionPolicies ?? null,
        modelType,
        vendorProfile,
        moeKernelPathProfile,
        hiddenActivation,
      });
    perfLog(`MoE L${layerIdx} expert_exec`, stepStart, { expertIdx, count });
  }

    const expertOutputsTensor = createTensor(
      expertOutputs,
      activationDtype,
      [numExperts, maxTokensPerExpert, hiddenSize],
      'moe_expert_outputs'
    );
    const perExpertScale = resolvePerExpertScaleBuffer(device, layerRouter?.perExpertScale || null);
    ownedPerExpertScaleBuffer = perExpertScale.ownedBuffer;
    stepStart = perfMark();
    outputTensor = await runScatterAddDynamic(
      expertOutputsTensor,
      indicesBuffer,
      weightsBuffer,
      tokenOffsets,
      numTokens,
      hiddenSize,
      topK,
      {
        weightsDtype: activationDtype,
        perExpertScale: perExpertScale.buffer,
      }
    );
  perfLog(`MoE L${layerIdx} scatter`, stepStart, { numTokens, hiddenSize });

    releaseBuffer(gathered.buffer);
    gathered = null;
    releaseBuffer(tokenMap);
    tokenMap = null;
    releaseBuffer(expertOutputs);
    expertOutputs = null;
    releaseBuffer(tokenOffsets);
    tokenOffsets = null;
    releaseBuffer(indicesBuffer);
    indicesBuffer = null;
    releaseBuffer(weightsBuffer);
    weightsBuffer = null;

    if (perfEnabled) {
      const cacheStats = getDequantCacheStats();
      trace.perf(`MoE L${layerIdx} done`, {
        numTokens,
        topK,
        executedExperts: scheduledExperts.length,
        activeExperts: scheduledExperts.length,
        activeExpertSelection: activeExpertSchedule?.selection ?? null,
        rowsPerExpert,
        maxTokensPerExpert,
        dequantCacheHits: cacheStats.hits,
        dequantCacheMisses: cacheStats.misses,
        expertCache: typeof expertLoader?.getExpertCacheStats === 'function'
          ? expertLoader.getExpertCacheStats()
          : null,
      });
    }

    return outputTensor.buffer;
  } finally {
    if (logitsBuffer) releaseBuffer(logitsBuffer);
    if (routerNormTensor?.buffer) releaseBuffer(routerNormTensor.buffer);
    if (routerScaledTensor?.buffer) releaseBuffer(routerScaledTensor.buffer);
    if (tokenCounts) releaseBuffer(tokenCounts);
    if (gathered?.buffer) releaseBuffer(gathered.buffer);
    if (tokenMap) releaseBuffer(tokenMap);
    if (expertOutputs) releaseBuffer(expertOutputs);
    if (tokenOffsets) releaseBuffer(tokenOffsets);
    if (indicesBuffer) releaseBuffer(indicesBuffer);
    if (weightsBuffer) releaseBuffer(weightsBuffer);
    if (ownedPerExpertScaleBuffer) releaseBuffer(ownedPerExpertScaleBuffer);
  }
}

function inferBufferDtype(buffer, expectedElements) {
  if (isWeightBuffer(buffer)) {
    return getWeightDtype(buffer);
  }
  const dtype = getWeightDtype(buffer);
  if (dtype) return dtype;
  const bytesPerElement = Math.round(buffer.size / expectedElements);
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });
}

function alignTo4(value) {
  return Math.ceil(value / 4) * 4;
}

function resolveMatrixStorageStrideBytes(buffer, rows, cols, label) {
  const dtype = inferBufferDtype(buffer, rows * cols, label);
  if (dtype === 'q4k') {
    return alignTo4(rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES);
  }
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  return alignTo4(rows * cols * bytesPerElement);
}

async function runGemma4RouteExperts({
  inputTensor,
  indicesBuffer,
  weightsBuffer,
  layerRouter,
  weights,
  expectedExpertFormat,
  profile,
  layerIdx,
  numTokens,
  topK,
  hiddenSize,
  intermediateSize,
  activationDtype,
  swigluLimit,
}) {
  const profileLabel = typeof profile?.label === 'string' && profile.label.length > 0
    ? profile.label
    : 'unknown MoE profile';
  if (activationDtype !== 'f16') {
    throw new Error(`[MoE] topk-route ${profileLabel} path requires f16 activations, got ${activationDtype}.`);
  }
  if (!weights?.gateUp || !weights?.down) {
    throw new Error(`[MoE] topk-route ${profileLabel} path missing packed weights for layer ${layerIdx}.`);
  }
  if (weights.expertFormat !== expectedExpertFormat) {
    throw new Error(
      `[MoE] topk-route ${profileLabel} expert format mismatch for layer ${layerIdx}: ` +
      `weights=${weights.expertFormat}, config=${expectedExpertFormat}`
    );
  }
  if (!layerRouter?.perExpertScale) {
    throw new Error(`[MoE] topk-route ${profileLabel} path requires per-expert router scale for layer ${layerIdx}.`);
  }

  const device = getDevice();
  const numRoutes = numTokens * topK;
  const gateUpOutDim = intermediateSize * 2;
  let gateUpOut = null;
  let activated = null;
  let routeDown = null;
  let ownedPerExpertScaleBuffer = null;

  try {
    gateUpOut = await runGemma4RouteQ4MatmulF16A(
      inputTensor,
      indicesBuffer,
      weights.gateUp,
      {
        numRoutes,
        topK,
        N: gateUpOutDim,
        K: hiddenSize,
        inputMode: 'token',
        label: `moe_l${layerIdx}_route_gate_up`,
      }
    );
    activated = await runSiLURowSplit(gateUpOut, {
      numTokens: numRoutes,
      dim: intermediateSize,
      activation: 'gelu',
      swigluLimit,
    });
    releaseBuffer(gateUpOut.buffer);
    gateUpOut = null;

    routeDown = await runGemma4RouteQ4MatmulF16A(
      activated,
      indicesBuffer,
      weights.down,
      {
        numRoutes,
        topK,
        N: hiddenSize,
        K: intermediateSize,
        inputMode: 'route',
        label: `moe_l${layerIdx}_route_down`,
      }
    );
    releaseBuffer(activated.buffer);
    activated = null;

    const perExpertScale = resolvePerExpertScaleBuffer(device, layerRouter.perExpertScale);
    ownedPerExpertScaleBuffer = perExpertScale.ownedBuffer;
    const outputTensor = await runScatterAddRoutesF16ExpertScale(
      routeDown,
      indicesBuffer,
      weightsBuffer,
      perExpertScale.buffer,
      numTokens,
      hiddenSize,
      topK,
      { label: `moe_l${layerIdx}_route_scatter` }
    );
    releaseBuffer(routeDown.buffer);
    routeDown = null;
    return outputTensor;
  } finally {
    if (gateUpOut?.buffer) releaseBuffer(gateUpOut.buffer);
    if (activated?.buffer) releaseBuffer(activated.buffer);
    if (routeDown?.buffer) releaseBuffer(routeDown.buffer);
    if (ownedPerExpertScaleBuffer) releaseBuffer(ownedPerExpertScaleBuffer);
  }
}

async function runGptOssExpert(
  gathered,
  expertOutputs,
  weights,
  layerIdx,
  expertIdx,
  count,
  inputOffset,
  outputOffset,
  hiddenSize,
  intermediateSize,
  numExperts,
  activationDtype,
  swigluLimit,
  kernelPath,
  executionPolicies,
  modelType,
  vendorProfile,
  moeKernelPathProfile
) {
  const perfEnabled = isTraceEnabled('perf');
  const perfMark = () => (perfEnabled ? performance.now() : 0);
  const perfLog = (label, start, data) => {
    if (!perfEnabled) return;
    trace.perf(`${label}: ${(performance.now() - start).toFixed(2)}ms`, data);
  };

  const outDim = intermediateSize * 2;

  const gateUpGroups = hiddenSize / 32;
  const downGroups = intermediateSize / 32;
  const totalExperts = weights.numExperts || numExperts;

  if (!weights.gateUpBlocks || !weights.gateUpScales || !weights.gateUpBias ||
      !weights.downBlocks || !weights.downScales) {
    const missing = [];
    if (!weights.gateUpBlocks) missing.push('gate_up_proj_blocks');
    if (!weights.gateUpScales) missing.push('gate_up_proj_scales');
    if (!weights.gateUpBias) missing.push('gate_up_proj_bias');
    if (!weights.downBlocks) missing.push('down_proj_blocks');
    if (!weights.downScales) missing.push('down_proj_scales');
    throw new Error(
      `[MoE] GPT-OSS expert ${expertIdx} missing tensors: ${missing.join(', ')}`
    );
  }

  let gateUpWeight;
  let downWeight;
  let stepStart = perfMark();
  const cached = getCachedDequant(layerIdx, expertIdx, activationDtype);

  if (cached) {
    gateUpWeight = cached.gateUp;
    downWeight = cached.down;
    perfLog(`MoE L${layerIdx} expert ${expertIdx} dequant_cache`, stepStart, { hit: true });
  } else {
    const gateUpTensor = await dequantizeMXFP4Expert(
      weights.gateUpBlocks,
      weights.gateUpScales,
      expertIdx,
      totalExperts,
      outDim,
      gateUpGroups,
      {
        outputDtype: activationDtype,
        modelType,
        groupSize: 32,
        dequantTileShape: vendorProfile.dequantTileShape,
      }
    );
    const downTensor = await dequantizeMXFP4Expert(
      weights.downBlocks,
      weights.downScales,
      expertIdx,
      totalExperts,
      hiddenSize,
      downGroups,
      {
        outputDtype: activationDtype,
        modelType,
        groupSize: 32,
        dequantTileShape: vendorProfile.dequantTileShape,
      }
    );
    gateUpWeight = gateUpTensor.buffer;
    downWeight = downTensor.buffer;
    setCachedDequant(layerIdx, expertIdx, activationDtype, gateUpWeight, downWeight);
    perfLog(`MoE L${layerIdx} expert ${expertIdx} dequant`, stepStart, {
      hit: false,
      dequantTileShape: vendorProfile.dequantTileShape,
      dequantKernel: moeKernelPathProfile?.dequantExpert ?? null,
    });
  }

  const gateUpOut = await runMatmul(
    gathered,
    gateUpWeight,
    count,
    outDim,
    hiddenSize,
    {
      transposeB: 'auto',
      aOffset: inputOffset,
      bDtype: activationDtype,
      outputDtype: activationDtype,
      role: 'moe_gate_up',
      kernelPath,
    }
  );

  const biasElements = totalExperts * outDim;
  const gateUpBiasDtype = inferBufferDtype(weights.gateUpBias, biasElements);
  let biasTensor = createTensor(weights.gateUpBias, gateUpBiasDtype, [biasElements], 'moe_gate_up_bias');
  let biasTemp = null;
  if (biasTensor.dtype !== activationDtype) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies,
      fromDtype: biasTensor.dtype,
      toDtype: activationDtype,
      op: 'moe_gate_up_bias',
      detail: `Expert ${expertIdx} gate/up bias would be repacked to match activation dtype.`,
    });
    biasTemp = activationDtype === 'f16'
      ? await castF32ToF16(biasTensor)
      : await castF16ToF32(biasTensor);
    biasTensor = biasTemp;
  }
  const biasBytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: biasTensor.dtype });
  const biasOffset = expertIdx * outDim * biasBytesPerElement;
  const activated = await runSwiGLURowsplitBias(
    gateUpOut,
    biasTensor,
    count,
    intermediateSize,
    { biasOffset, swigluLimit }
  );
  if (biasTemp) {
    releaseBuffer(biasTemp.buffer);
  }
  releaseBuffer(gateUpOut.buffer);

  await runMatmul(
    activated,
    downWeight,
    count,
    hiddenSize,
    intermediateSize,
    {
      transposeB: 'auto',
      outputBuffer: expertOutputs,
      cOffset: outputOffset,
      bDtype: activationDtype,
      outputDtype: activationDtype,
      role: 'moe_down',
      kernelPath,
    }
  );
  releaseBuffer(activated.buffer);

  if (weights.downBias) {
    const biasElements = totalExperts * hiddenSize;
    const downBiasDtype = inferBufferDtype(weights.downBias, biasElements);
    const downBiasBytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
    const downBiasOffset = expertIdx * hiddenSize * downBiasBytesPerElement;
    const expertOutputsTensor = createTensor(expertOutputs, activationDtype, [count, hiddenSize], 'expert_outputs');
    const downBiasTensor = createTensor(weights.downBias, downBiasDtype, [biasElements], 'down_bias');
    await runBiasAdd(expertOutputsTensor, downBiasTensor, count, hiddenSize, {
      dataOffset: outputOffset,
      biasOffset: downBiasOffset,
    });
  }
}

async function runGemma4Expert(
  gathered,
  expertOutputs,
  weights,
  count,
  inputOffset,
  outputOffset,
  hiddenSize,
  intermediateSize,
  activationDtype,
  swigluLimit,
  kernelPath
) {
  const numExperts = weights.numExperts;
  const expertIdx = weights.expertIdx;
  if (!Number.isFinite(numExperts) || numExperts <= 0) {
    throw new Error(`[MoE] Gemma-style expert ${expertIdx} missing numExperts.`);
  }
  if (!Number.isFinite(expertIdx) || expertIdx < 0) {
    throw new Error('[MoE] Gemma-style expert missing expertIdx.');
  }
  if (expertIdx >= numExperts) {
    throw new Error(`[MoE] Gemma-style expert index ${expertIdx} out of range for ${numExperts} experts.`);
  }

  const gateUpOutDim = intermediateSize * 2;
  const gateUpStrideBytes = resolveMatrixStorageStrideBytes(
    weights.gateUp,
    gateUpOutDim,
    hiddenSize,
    'Gemma gate_up_proj'
  );
  const gateUpOffset = expertIdx * gateUpStrideBytes;
  const downStrideBytes = resolveMatrixStorageStrideBytes(
    weights.down,
    hiddenSize,
    intermediateSize,
    'Gemma down_proj'
  );
  const downOffset = expertIdx * downStrideBytes;

  const gateUpOut = await runMatmul(
    gathered,
    weights.gateUp,
    count,
    gateUpOutDim,
    hiddenSize,
    {
      transposeB: true,
      aOffset: inputOffset,
      bOffset: gateUpOffset,
      outputDtype: activationDtype,
      role: 'moe_gate_up',
      kernelPath,
    }
  );

  const activated = await runSiLURowSplit(gateUpOut, {
    numTokens: count,
    dim: intermediateSize,
    activation: 'gelu',
    swigluLimit,
  });
  releaseBuffer(gateUpOut.buffer);

  await runMatmul(
    activated,
    weights.down,
    count,
    hiddenSize,
    intermediateSize,
    {
      transposeB: true,
      bOffset: downOffset,
      outputBuffer: expertOutputs,
      cOffset: outputOffset,
      outputDtype: activationDtype,
      role: 'moe_down',
      kernelPath,
    }
  );
  releaseBuffer(activated.buffer);
}

async function runMixtralExpert(
  gathered,
  expertOutputs,
  weights,
  count,
  inputOffset,
  outputOffset,
  hiddenSize,
  intermediateSize,
  hiddenActivation,
  activationDtype,
  swigluLimit,
  kernelPath
) {
  const gateOut = await runMatmul(
    gathered,
    weights.gate,
    count,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      aOffset: inputOffset,
      outputDtype: activationDtype,
      role: 'moe_gate',
      kernelPath,
    }
  );
  const upOut = await runMatmul(
    gathered,
    weights.up,
    count,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      aOffset: inputOffset,
      outputDtype: activationDtype,
      role: 'moe_up',
      kernelPath,
    }
  );

  const activationFn = {
    gelu: runGeLU,
    silu: runSiLU,
  }[selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation })];
  const activated = await activationFn(upOut, {
    size: count * intermediateSize,
    gate: gateOut,
    swigluLimit,
  });
  releaseBuffer(gateOut.buffer);
  releaseBuffer(upOut.buffer);

  await runMatmul(
    activated,
    weights.down,
    count,
    hiddenSize,
    intermediateSize,
    {
      transposeB: 'auto',
      outputBuffer: expertOutputs,
      cOffset: outputOffset,
      outputDtype: activationDtype,
      role: 'moe_down',
      kernelPath,
    }
  );
  releaseBuffer(activated.buffer);
}
