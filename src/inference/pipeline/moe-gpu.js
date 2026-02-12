import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../../gpu/tensor.js';
import { castF16ToF32, castF32ToF16 } from '../../gpu/kernels/cast.js';
import {
  runMatmul,
  runSiLU,
  runGeLU,
  dequantizeMXFP4Expert,
  runBiasAdd,
  runSoftmaxTopK,
  runMoEGather,
  runMoEBuildTokenOffsets,
  runScatterAddDynamic,
  runSwiGLURowsplitBias,
} from '../../gpu/kernel-selector.js';
import { trace, isTraceEnabled } from '../../debug/index.js';
import { f16ToF32Array } from '../kv-cache/types.js';
import { resolveMaxTokensPerExpert, getCachedDequant, setCachedDequant, getDequantCacheStats } from './moe-cache.js';
import { ensureExpertLoaded } from './moe-helpers.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

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

  const { hiddenSize, numExperts, intermediateSize, moeTopK, hiddenActivation } = config;
  const expertFormat = config.expertFormat;
  const swigluLimit = config.swigluLimit;
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

  const layerRouter = layerRouterWeights?.get(layerIdx) || null;
  if (layerRouter) {
    moeRouter.loadWeights(layerRouter.weight, layerRouter.bias || null);
  }

  let stepStart = perfMark();
  const logitsBuffer = await moeRouter.computeRouterLogitsGPU(inputTensor.buffer, numTokens, null, {
    inputDtype: activationDtype,
    outputDtype: activationDtype,
  });
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

  stepStart = perfMark();
  const { indices: indicesBuffer, weights: weightsBuffer } = await runSoftmaxTopK(
    logitsBuffer,
    numTokens,
    numExperts,
    topK,
    { normalize: moeRouter.normalizeWeights, inputDtype: logitsDtype, weightsDtype: activationDtype }
  );
  perfLog(`MoE L${layerIdx} topk`, stepStart, { topK });

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

  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
  const bytesPerToken = hiddenSize * bytesPerElement;
  let maxTokensPerExpert = resolveMaxTokensPerExpert(numTokens, numExperts, topK, hiddenSize, activationDtype);

  let gathered;
  let tokenCounts;
  let tokenMap;

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

  const expertOutputs = acquireBuffer(
    numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement,
    undefined,
    'moe_expert_outputs_gathered'
  );

  const zeroEncoder = device.createCommandEncoder({ label: 'zero_moe_expert_outputs' });
  zeroEncoder.clearBuffer(expertOutputs, 0, numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement);
  device.queue.submit([zeroEncoder.finish()]);

  stepStart = perfMark();
  const tokenOffsets = await runMoEBuildTokenOffsets(
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

  const expertStrideBytes = maxTokensPerExpert * bytesPerToken;
  const rowsPerExpert = maxTokensPerExpert;

  // GPU-first execution path: avoid CPU readback of tokenCounts for scheduling.
  // Each expert executes with a fixed row budget (maxTokensPerExpert); gathered
  // unused rows are zero-filled and never consumed by scatter_add_dynamic.
  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = rowsPerExpert;

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

    const inputOffset = expertIdx * expertStrideBytes;
    const outputOffset = expertIdx * expertStrideBytes;

    stepStart = perfMark();
    if (weights.expertFormat !== expertFormat) {
      throw new Error(
        `[MoE] Expert format mismatch for ${expertKey}: ` +
        `weights=${weights.expertFormat}, config=${expertFormat}`
      );
    }

    if (expertFormat === 'gpt-oss') {
      await runGptOssExpert(
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
        swigluLimit
      );
    } else if (expertFormat === 'mixtral' && weights.gate && weights.up && weights.down) {
      await runMixtralExpert(
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
        swigluLimit
      );
    } else if (expertFormat === 'mixtral') {
      throw new Error(`[MoE] Missing Mixtral weights for ${expertKey}`);
    }
    perfLog(`MoE L${layerIdx} expert_exec`, stepStart, { expertIdx, count });
  }

  const expertOutputsTensor = createTensor(
    expertOutputs,
    activationDtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_expert_outputs'
  );
  stepStart = perfMark();
  const outputTensor = await runScatterAddDynamic(
    expertOutputsTensor,
    indicesBuffer,
    weightsBuffer,
    tokenOffsets,
    numTokens,
    hiddenSize,
    topK,
    { weightsDtype: activationDtype }
  );
  perfLog(`MoE L${layerIdx} scatter`, stepStart, { numTokens, hiddenSize });

  releaseBuffer(gathered.buffer);
  releaseBuffer(tokenMap);
  releaseBuffer(expertOutputs);
  releaseBuffer(tokenOffsets);
  releaseBuffer(indicesBuffer);
  releaseBuffer(weightsBuffer);

  if (perfEnabled) {
    const cacheStats = getDequantCacheStats();
    trace.perf(`MoE L${layerIdx} done`, {
      numTokens,
      topK,
      executedExperts: numExperts,
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
}

function inferBufferDtype(buffer, expectedElements) {
  const bytesPerElement = Math.round(buffer.size / expectedElements);
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });
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
  swigluLimit
) {
  const perfEnabled = isTraceEnabled('perf');
  const perfMark = () => (perfEnabled ? performance.now() : 0);
  const perfLog = (label, start, data) => {
    if (!perfEnabled) return;
    trace.perf(`${label}: ${(performance.now() - start).toFixed(2)}ms`, data);
  };

  const outDim = intermediateSize * 2;

  if (hiddenSize % 32 !== 0 || intermediateSize % 32 !== 0) {
    throw new Error(
      `[MoE] GPT-OSS MXFP4 expects hiddenSize and intermediateSize divisible by 32, got ` +
      `hiddenSize=${hiddenSize} intermediateSize=${intermediateSize}`
    );
  }

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
      { outputDtype: activationDtype }
    );
    const downTensor = await dequantizeMXFP4Expert(
      weights.downBlocks,
      weights.downScales,
      expertIdx,
      totalExperts,
      hiddenSize,
      downGroups,
      { outputDtype: activationDtype }
    );
    gateUpWeight = gateUpTensor.buffer;
    downWeight = downTensor.buffer;
    setCachedDequant(layerIdx, expertIdx, activationDtype, gateUpWeight, downWeight);
    perfLog(`MoE L${layerIdx} expert ${expertIdx} dequant`, stepStart, { hit: false });
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
    }
  );

  const biasElements = totalExperts * outDim;
  const gateUpBiasDtype = inferBufferDtype(weights.gateUpBias, biasElements);
  let biasTensor = createTensor(weights.gateUpBias, gateUpBiasDtype, [biasElements], 'moe_gate_up_bias');
  let biasTemp = null;
  if (biasTensor.dtype !== activationDtype) {
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
  swigluLimit
) {
  const gateOut = await runMatmul(
    gathered,
    weights.gate,
    count,
    intermediateSize,
    hiddenSize,
    { transposeB: 'auto', aOffset: inputOffset, outputDtype: activationDtype, role: 'moe_gate' }
  );
  const upOut = await runMatmul(
    gathered,
    weights.up,
    count,
    intermediateSize,
    hiddenSize,
    { transposeB: 'auto', aOffset: inputOffset, outputDtype: activationDtype, role: 'moe_up' }
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
    { transposeB: 'auto', outputBuffer: expertOutputs, cOffset: outputOffset, outputDtype: activationDtype, role: 'moe_down' }
  );
  releaseBuffer(activated.buffer);
}
