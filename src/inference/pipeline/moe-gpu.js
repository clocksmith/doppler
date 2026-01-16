import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
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
  runScatterAddDynamic,
  runSwiGLURowsplitBias,
} from '../../gpu/kernel-selector.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { f16ToF32Array } from '../kv-cache/types.js';
import { resolveMaxTokensPerExpert, getCachedDequant, setCachedDequant, getDequantCacheStats } from './moe-cache.js';
import { ensureExpertLoaded } from './moe-helpers.js';

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
  const topK = moeTopK || moeRouter.topK || 2;
  const activationDtype = config.activationDtype === 'f16' ? 'f16' : 'f32';

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
    const logitsBytesPerElement = logitsDtype === 'f16' ? 2 : 4;
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

    const weightsBytes = numTokens * topK * (activationDtype === 'f16' ? 2 : 4);
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

  const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
  const bytesPerToken = hiddenSize * bytesPerElement;
  let maxTokensPerExpert = resolveMaxTokensPerExpert(numTokens, numExperts, topK, hiddenSize, activationDtype);

  let gathered;
  let tokenCounts;
  let tokenMap;
  let tokenCountsCPU;
  let tokenMapCPU;
  let gatherAttempts = 0;

  while (true) {
    gatherAttempts += 1;
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
    perfLog(`MoE L${layerIdx} gather`, stepStart, { maxTokensPerExpert, attempt: gatherAttempts });

    stepStart = perfMark();
    const countsData = await readBuffer(tokenCounts, numExperts * 4);
    tokenCountsCPU = new Uint32Array(countsData);
    let maxCount = 0;
    let totalCount = 0;
    for (let i = 0; i < tokenCountsCPU.length; i++) {
      const v = tokenCountsCPU[i];
      totalCount += v;
      if (v > maxCount) maxCount = v;
    }
    perfLog(`MoE L${layerIdx} counts_readback`, stepStart, { maxCount });
    if (isTraceEnabled('buffers')) {
      let minCount = Number.MAX_SAFE_INTEGER;
      let zeroCount = 0;
      let overMax = 0;
      for (let i = 0; i < tokenCountsCPU.length; i++) {
        const v = tokenCountsCPU[i];
        if (v < minCount) minCount = v;
        if (v === 0) zeroCount += 1;
        if (v > maxTokensPerExpert) overMax += 1;
      }
      trace.buffers(`MoE L${layerIdx} token_counts`, {
        minCount,
        maxCount,
        zeroCount,
        overMax,
        totalCount,
        expected: numTokens * topK,
        sample: Array.from(tokenCountsCPU.slice(0, 8)),
      });
    }

    if (maxCount > maxTokensPerExpert) {
      releaseBuffer(gathered.buffer);
      releaseBuffer(tokenCounts);
      releaseBuffer(tokenMap);

      if (maxTokensPerExpert >= numTokens) {
        throw new Error(
          `[MoE] Gather overflow: maxCount=${maxCount} > maxTokensPerExpert=${maxTokensPerExpert}`
        );
      }

      const expanded = Math.ceil(Math.max(maxCount * 1.2, maxTokensPerExpert * 2));
      maxTokensPerExpert = expanded;
      if (activationDtype === 'f16') {
        const bytesPerTokenAligned = hiddenSize * 2;
        const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
        const alignMultiple = 256 / gcd(256, bytesPerTokenAligned);
        maxTokensPerExpert = Math.ceil(maxTokensPerExpert / alignMultiple) * alignMultiple;
      } else {
        maxTokensPerExpert = Math.min(maxTokensPerExpert, numTokens);
      }
      if (perfEnabled) {
        trace.perf(`MoE L${layerIdx} gather_retry -> ${maxTokensPerExpert}`);
      }
      continue;
    }

    stepStart = perfMark();
    const tokenMapElems = numExperts * maxTokensPerExpert * 2;
    const tokenMapData = await readBuffer(tokenMap, tokenMapElems * 4);
    tokenMapCPU = new Uint32Array(tokenMapData);
    perfLog(`MoE L${layerIdx} map_readback`, stepStart, { tokenMapElems });
    break;
  }

  const expertOutputs = acquireBuffer(
    numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement,
    undefined,
    'moe_expert_outputs_gathered'
  );

  const zeroEncoder = device.createCommandEncoder({ label: 'zero_moe_expert_outputs' });
  zeroEncoder.clearBuffer(expertOutputs, 0, numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement);
  device.queue.submit([zeroEncoder.finish()]);

  stepStart = perfMark();
  const tokenOffsetsCPU = new Uint32Array(numTokens * topK);
  tokenOffsetsCPU.fill(0xFFFFFFFF);

  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCountsCPU[expertIdx] || 0;
    if (count > maxTokensPerExpert) {
      throw new Error(
        `[MoE] Gather overflow: expert ${expertIdx} count=${count} > maxTokensPerExpert=${maxTokensPerExpert}`
      );
    }
    for (let slotIdx = 0; slotIdx < count; slotIdx++) {
      const mapBase = (expertIdx * maxTokensPerExpert + slotIdx) * 2;
      const tokenIdx = tokenMapCPU[mapBase];
      const kIdx = tokenMapCPU[mapBase + 1];
      tokenOffsetsCPU[tokenIdx * topK + kIdx] = expertIdx * maxTokensPerExpert + slotIdx;
    }
  }
  perfLog(`MoE L${layerIdx} offsets_build`, stepStart);

  for (let i = 0; i < tokenOffsetsCPU.length; i++) {
    if (tokenOffsetsCPU[i] === 0xFFFFFFFF) {
      const tokenIdx = Math.floor(i / topK);
      const kIdx = i % topK;
      log.error('MoE', `Missing offset at i=${i} (token=${tokenIdx}, k=${kIdx})`);
      throw new Error(`[MoE] tokenOffsets incomplete at i=${i}`);
    }
  }

  stepStart = perfMark();
  const tokenOffsets = acquireBuffer(tokenOffsetsCPU.byteLength, undefined, 'moe_token_offsets');
  device.queue.writeBuffer(tokenOffsets, 0, tokenOffsetsCPU);
  perfLog(`MoE L${layerIdx} offsets_upload`, stepStart, { bytes: tokenOffsetsCPU.byteLength });

  releaseBuffer(tokenCounts);

  const expertStrideBytes = maxTokensPerExpert * bytesPerToken;
  const activeExperts = [];
  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCountsCPU[expertIdx] || 0;
    if (count > 0) activeExperts.push(expertIdx);
  }

  if (typeof expertLoader?.predictNextLayerExperts === 'function' &&
      typeof expertLoader?.prefetchExperts === 'function') {
    const predicted = expertLoader.predictNextLayerExperts(activeExperts);
    if (predicted?.length) {
      expertLoader.prefetchExperts(layerIdx + 1, predicted);
    }
  }

  for (const expertIdx of activeExperts) {
    const count = tokenCountsCPU[expertIdx] || 0;
    if (count === 0) continue;

    stepStart = perfMark();
    await ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader);
    perfLog(`MoE L${layerIdx} expert_load`, stepStart, { expertIdx, count });
    const expertKey = `layer_${layerIdx}_expert_${expertIdx}`;
    const weights = expertWeights.get(expertKey);
    if (!weights) continue;

    const inputOffset = expertIdx * expertStrideBytes;
    const outputOffset = expertIdx * expertStrideBytes;

    stepStart = perfMark();
    if (weights.isGptOss) {
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
        config.swigluLimit ?? null
      );
    } else if (weights.gate && weights.up && weights.down) {
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
        config.swigluLimit ?? null
      );
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
      activeExperts: activeExperts.length,
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
  return bytesPerElement <= 2 ? 'f16' : 'f32';
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
    log.warn('MoE', `GPT-OSS expert ${expertIdx} missing tensors, skipping`);
    return;
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
  const biasOffset = expertIdx * outDim * (biasTensor.dtype === 'f16' ? 2 : 4);
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
    const downBiasOffset = expertIdx * hiddenSize * (activationDtype === 'f16' ? 2 : 4);
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

  const activationFn = hiddenActivation === 'gelu' ? runGeLU : runSiLU;
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
