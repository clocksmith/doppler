import { getRuntimeConfig } from '../../../config/runtime.js';
import { getDevice } from '../../../gpu/device.js';
import { createWeightBuffer, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { Q4K_BLOCK_BYTES, QK_K } from '../../../config/schema/index.js';

function shapeMatches(weight, rows, cols) {
  return Array.isArray(weight?.shape)
    && weight.shape.length === 2
    && Number(weight.shape[0]) === rows
    && Number(weight.shape[1]) === cols;
}

function canUseLinearAttentionABProjectionFusion({
  phase,
  numTokens,
  debugProbes,
  operatorDiagnostics,
}) {
  const session = getRuntimeConfig()?.inference?.session;
  if (session?.useLinearAttentionABProjectionFusion !== true) {
    return false;
  }
  if (phase !== 'decode' || numTokens !== 1) {
    return false;
  }
  if (debugProbes?.length || operatorDiagnostics?.enabled === true) {
    return false;
  }
  return true;
}

export function resolveLinearAttentionABProjection(layerWeights, options) {
  const {
    phase,
    numTokens,
    hiddenSize,
    numVHeads,
    layerIdx,
    debugProbes,
    operatorDiagnostics,
  } = options;
  if (!canUseLinearAttentionABProjectionFusion({
    phase,
    numTokens,
    debugProbes,
    operatorDiagnostics,
  })) {
    return null;
  }
  if (layerWeights.linearABProj && shapeMatches(layerWeights.linearABProj, numVHeads * 2, hiddenSize)) {
    return {
      weight: layerWeights.linearABProj,
      outDim: numVHeads * 2,
      bProjOffsetElements: numTokens * numVHeads,
    };
  }

  const aWeight = layerWeights.linearInProjA;
  const bWeight = layerWeights.linearInProjB;
  if (
    !isWeightBuffer(aWeight)
    || !isWeightBuffer(bWeight)
    || aWeight.dtype !== 'f16'
    || bWeight.dtype !== 'f16'
    || aWeight.layout !== 'row'
    || bWeight.layout !== 'row'
    || !shapeMatches(aWeight, numVHeads, hiddenSize)
    || !shapeMatches(bWeight, numVHeads, hiddenSize)
  ) {
    return null;
  }

  const bytesPerWeight = numVHeads * hiddenSize * 2;
  if (aWeight.buffer.size < bytesPerWeight || bWeight.buffer.size < bytesPerWeight) {
    return null;
  }
  const device = getDevice();
  if (!device) {
    return null;
  }

  const buffer = device.createBuffer({
    label: `L${layerIdx}.linear_ab_proj_weight`,
    size: bytesPerWeight * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder({ label: `L${layerIdx}.linear_ab_proj_weight_pack` });
  encoder.copyBufferToBuffer(aWeight.buffer, 0, buffer, 0, bytesPerWeight);
  encoder.copyBufferToBuffer(bWeight.buffer, 0, buffer, bytesPerWeight, bytesPerWeight);
  device.queue.submit([encoder.finish()]);

  layerWeights.linearABProj = createWeightBuffer(
    buffer,
    'f16',
    'row',
    [numVHeads * 2, hiddenSize],
    `L${layerIdx}.linear_ab_proj_weight`
  );
  return {
    weight: layerWeights.linearABProj,
    outDim: numVHeads * 2,
    bProjOffsetElements: numTokens * numVHeads,
  };
}

function canUseLinearAttentionQKVZProjectionFusion({
  phase,
  numTokens,
  debugProbes,
  operatorDiagnostics,
}) {
  const session = getRuntimeConfig()?.inference?.session;
  if (session?.useLinearAttentionQKVZProjectionFusion !== true) {
    return false;
  }
  if (phase !== 'decode' || numTokens !== 1) {
    return false;
  }
  if (debugProbes?.length || operatorDiagnostics?.enabled === true) {
    return false;
  }
  return true;
}

function q4kRowBytes(hiddenSize) {
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    return null;
  }
  return Math.ceil(hiddenSize / QK_K) * Q4K_BLOCK_BYTES;
}

export function resolveLinearAttentionQKVZProjection(layerWeights, options) {
  const {
    phase,
    numTokens,
    hiddenSize,
    convDim,
    valueDim,
    layerIdx,
    debugProbes,
    operatorDiagnostics,
  } = options;
  if (!canUseLinearAttentionQKVZProjectionFusion({
    phase,
    numTokens,
    debugProbes,
    operatorDiagnostics,
  })) {
    return null;
  }
  if (layerWeights.linearQKVZProj && shapeMatches(layerWeights.linearQKVZProj, convDim + valueDim, hiddenSize)) {
    return {
      weight: layerWeights.linearQKVZProj,
      outDim: convDim + valueDim,
    };
  }

  const qkvWeight = layerWeights.qkvProj;
  const zWeight = layerWeights.linearInProjZ;
  if (
    !isWeightBuffer(qkvWeight)
    || !isWeightBuffer(zWeight)
    || qkvWeight.dtype !== 'q4k'
    || zWeight.dtype !== 'q4k'
    || qkvWeight.layout !== 'row'
    || zWeight.layout !== 'row'
    || !shapeMatches(qkvWeight, convDim, hiddenSize)
    || !shapeMatches(zWeight, valueDim, hiddenSize)
  ) {
    return null;
  }

  const rowBytes = q4kRowBytes(hiddenSize);
  if (rowBytes == null) {
    return null;
  }
  const qkvBytes = convDim * rowBytes;
  const zBytes = valueDim * rowBytes;
  if (qkvWeight.buffer.size < qkvBytes || zWeight.buffer.size < zBytes) {
    return null;
  }
  const device = getDevice();
  if (!device) {
    return null;
  }

  const buffer = device.createBuffer({
    label: `L${layerIdx}.linear_qkvz_proj_weight`,
    size: qkvBytes + zBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder({ label: `L${layerIdx}.linear_qkvz_proj_weight_pack` });
  encoder.copyBufferToBuffer(qkvWeight.buffer, 0, buffer, 0, qkvBytes);
  encoder.copyBufferToBuffer(zWeight.buffer, 0, buffer, qkvBytes, zBytes);
  device.queue.submit([encoder.finish()]);

  layerWeights.linearQKVZProj = createWeightBuffer(
    buffer,
    'q4k',
    'row',
    [convDim + valueDim, hiddenSize],
    `L${layerIdx}.linear_qkvz_proj_weight`
  );
  return {
    weight: layerWeights.linearQKVZProj,
    outDim: convDim + valueDim,
  };
}
