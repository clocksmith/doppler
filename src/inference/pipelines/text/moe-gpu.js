import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  runRMSNorm,
  runScale,
  runSoftmaxTopK,
  runMoEGather,
  runMoEBuildTokenOffsets,
  runScatterAddDynamic,
} from '../../../gpu/kernel-selector.js';
import { trace, isTraceEnabled } from '../../../debug/index.js';
import { f16ToF32Array } from '../../kv-cache/types.js';
import { resolveMaxTokensPerExpert, getDequantCacheStats } from './moe-cache.js';
import { ensureExpertLoaded } from './moe/expert-loading.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import {
  validateMoeShape,
  resolveMoeExecutionProfile,
  resolveMoeIntermediateSize,
  resolveMoeVendorProfile,
  resolveMoeKernelPathProfile,
} from './moe-shape-validator.js';
import {
  resolvePerExpertScaleBuffer,
  runGemma4Expert,
  runGemma4RouteExperts,
  runGptOssExpert,
  runMixtralExpert,
} from './moe/gpu-executor.js';
import {
  buildActiveExpertScheduleFromIndices,
  resolveMoEActiveExpertSelection,
} from './moe/plan.js';

export { buildActiveExpertScheduleFromIndices } from './moe/plan.js';

function resolveConfiguredMoEActiveExpertSelection() {
  return resolveMoEActiveExpertSelection(
    getRuntimeConfig()?.inference?.moe?.routing?.activeExpertSelection
  );
}

async function resolveActiveExpertSchedule(indicesBuffer, numTokens, numExperts, topK, maxTokensPerExpert) {
  const selection = resolveConfiguredMoEActiveExpertSelection();
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

  const activeExpertSelection = resolveConfiguredMoEActiveExpertSelection();
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
