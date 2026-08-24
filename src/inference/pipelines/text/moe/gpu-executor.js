import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, BufferUsage, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { castF16ToF32, castF32ToF16 } from '../../../../gpu/kernels/cast.js';
import {
  runMatmul,
  runSiLU,
  runSiLURowSplit,
  runGeLU,
  dequantizeMXFP4Expert,
  runBiasAdd,
  runSwiGLURowsplitBias,
  runGemma4RouteQ4MatmulF16A,
  runScatterAddRoutesF16ExpertScale,
} from '../../../../gpu/kernel-selector.js';
import {
  getBuffer,
  getWeightDtype,
  isGpuBufferInstance,
  isWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import { trace, isTraceEnabled } from '../../../../debug/index.js';
import { getCachedDequant, setCachedDequant } from '../moe-cache.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../../config/schema/index.js';

export function resolvePerExpertScaleBuffer(device, value) {
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

export async function runGemma4RouteExperts({
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

export async function runGptOssExpert(
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

export async function runGemma4Expert(
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

export async function runMixtralExpert(
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
    inputActivation: 'identity',
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


