import { getDevice } from '../../../../gpu/device.js';
import { createTrainingConfig } from '../../../../config/training-defaults.js';
import {
  runAttention,
  castF16ToF32,
  runGather,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runRoPE,
  runScale,
  runSiLU,
  runSiLURowSplit,
} from '../../../../gpu/kernels/index.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { OpType } from '../../autograd.js';
import { LoraAdapter } from '../../lora.js';
import { normalizeOptionalString } from '../suite-data.js';
import { LORA_MODULE_ALIASES } from '../../../../inference/pipelines/text/lora.js';
import { resolveEmbeddingScale } from '../../../../inference/pipelines/text/embed.js';
import { isSlidingLayerType } from '../../../../inference/pipelines/text/attention/dispatch-params.js';
import { createDistillStudentProjectionModelFixture, ensureTrainableTensor, normalizeTransformerLoraConfig, releaseTensor, resolvePhasePrompts, resolveTensorDtype } from './model.js';

export const DISTILL_STUDENT_GRAPH_PROJECTION = 'projection_head';

export const DISTILL_STUDENT_GRAPH_FULL = 'transformer_full';

export function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tokens');
}

export function tensorElementCount(shape) {
  return shape.reduce((product, value) => product * value, 1);
}

export async function recordTensorView(tape, input, shape, label) {
  if (tensorElementCount(input.shape) !== tensorElementCount(shape)) {
    throw new Error(`${label} cannot change tensor element count.`);
  }
  return tape.record(
    OpType.RESHAPE,
    (value) => createTensor(value.buffer, value.dtype, [...shape], label),
    [input],
    { shape: [...shape] }
  );
}

export function normalizeDistillStudentGraphMode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return DISTILL_STUDENT_GRAPH_FULL;
  const compact = normalized.toLowerCase().replace(/[-\s]/g, '_');
  if (compact === 'projection_head' || compact === 'projection') {
    return DISTILL_STUDENT_GRAPH_PROJECTION;
  }
  return DISTILL_STUDENT_GRAPH_FULL;
}

export function resolveTransformerLoraShape(moduleName, dims) {
  if (moduleName === 'q_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.numHeads * dims.headDim };
  }
  if (moduleName === 'k_proj' || moduleName === 'v_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.numKVHeads * dims.headDim };
  }
  if (moduleName === 'o_proj') {
    return { inDim: dims.attentionSize, outDim: dims.hiddenSize };
  }
  if (moduleName === 'gate_up_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.intermediateSize * 2 };
  }
  if (moduleName === 'gate_proj' || moduleName === 'up_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.intermediateSize };
  }
  if (moduleName === 'down_proj') {
    return { inDim: dims.intermediateSize, outDim: dims.hiddenSize };
  }
  throw new Error(`Transformer LoRA target module "${moduleName}" is not supported.`);
}

export function createTransformerLoraAdapters(config, dims) {
  if (!config) return {};
  const adapters = {};
  for (const moduleName of config.targetModules) {
    const shape = resolveTransformerLoraShape(moduleName, dims);
    adapters[moduleName] = new LoraAdapter({
      inDim: shape.inDim,
      outDim: shape.outDim,
      rank: config.rank,
      alpha: config.alpha,
      dtype: 'f32',
    });
  }
  return adapters;
}

export function disposeTransformerLoraAdapters(layers) {
  for (const layer of layers) {
    for (const adapter of Object.values(layer.lora || {})) {
      if (adapter && typeof adapter.dispose === 'function') {
        adapter.dispose();
      }
    }
  }
}

export function getTensorRows(value) {
  const shape = Array.isArray(value?.shape) ? value.shape : null;
  const rows = Number(shape?.[0]);
  return Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : null;
}

export function getTensorCols(value) {
  const shape = Array.isArray(value?.shape) ? value.shape : null;
  const cols = Number(shape?.[1]);
  return Number.isFinite(cols) && cols > 0 ? Math.floor(cols) : null;
}

export function resolveLayerFfnIntermediateSize(layerIdx, weights, fallback) {
  const gateUpRows = getTensorRows(weights.gateUp || weights.ffnGateUp);
  if (gateUpRows !== null && gateUpRows % 2 === 0) {
    return gateUpRows / 2;
  }
  const gateRows = getTensorRows(weights.gate || weights.ffnGate);
  if (gateRows !== null) {
    return gateRows;
  }
  const upRows = getTensorRows(weights.up || weights.ffnUp);
  if (upRows !== null) {
    return upRows;
  }
  const downCols = getTensorCols(weights.down || weights.ffnDown);
  if (downCols !== null) {
    return downCols;
  }
  if (Number.isInteger(fallback) && fallback > 0) {
    return fallback;
  }
  throw new Error(`Distill full-graph student cannot resolve FFN size for layer ${layerIdx}.`);
}

export function createRowSliceTensor(inputTensor, rows, cols, rowIndex, label) {
  const device = getDevice();
  if (!device) {
    throw new Error('Distill full-graph student requires active GPU device.');
  }
  const dtype = inputTensor?.dtype === 'f16' ? 'f16' : 'f32';
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const rowBytes = cols * bytesPerElement;
  const clampedRow = Math.max(0, Math.min(rows - 1, rowIndex));
  const outputBuffer = acquireBuffer(rowBytes, undefined, label);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    inputTensor.buffer,
    clampedRow * rowBytes,
    outputBuffer,
    0,
    rowBytes
  );
  device.queue.submit([encoder.finish()]);
  return createTensor(outputBuffer, dtype, [1, cols], label);
}
