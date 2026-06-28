import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import {
  assertRMSNormWeightBuffer,
  planRMSNormDispatch,
  resolveNormWeightDtype,
} from './rmsnorm.js';

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`RMSNorm QK requires ${label} to be a positive integer.`);
  }
}

function assertBufferCapacity(buffer, requiredBytes, label) {
  if (Number.isFinite(buffer?.size) && requiredBytes > buffer.size) {
    throw new Error(`RMSNorm QK ${label} buffer is smaller than requested range (${requiredBytes} > ${buffer.size} bytes).`);
  }
}

function selectRMSNormQKVariant(dtype) {
  return selectRuleValue('rmsnormQk', 'variant', { isF16: dtype === 'f16' });
}

export function canUseRMSNormQK(qTensor, kTensor, options = {}) {
  if (options.skipKNorm === true) {
    return false;
  }
  if (!qTensor || !kTensor || qTensor.buffer === kTensor.buffer) {
    return false;
  }
  return qTensor.dtype === kTensor.dtype && (qTensor.dtype === 'f32' || qTensor.dtype === 'f16');
}

async function rmsNormQK(target, qTensor, kTensor, qWeight, kWeight, eps, options = {}) {
  const {
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
    rmsNormWeightOffset = false,
  } = options;

  assertPositiveInteger(numTokens, 'numTokens');
  assertPositiveInteger(numHeads, 'numHeads');
  assertPositiveInteger(numKVHeads, 'numKVHeads');
  assertPositiveInteger(headDim, 'headDim');
  if (qTensor.dtype !== kTensor.dtype) {
    throw new Error(`RMSNorm QK requires matching Q/K dtypes, got ${qTensor.dtype} and ${kTensor.dtype}.`);
  }
  if (qTensor.dtype !== 'f32' && qTensor.dtype !== 'f16') {
    throw new Error(`RMSNorm QK requires f32 or f16 tensors, got ${qTensor.dtype}.`);
  }

  const qRows = numTokens * numHeads;
  const kRows = numTokens * numKVHeads;
  const bytesPerElement = dtypeBytes(qTensor.dtype);
  assertBufferCapacity(qTensor.buffer, qRows * headDim * bytesPerElement, 'Q input');
  assertBufferCapacity(kTensor.buffer, kRows * headDim * bytesPerElement, 'K input');

  const qWeightBuffer = getBuffer(qWeight);
  const kWeightBuffer = getBuffer(kWeight);
  assertRMSNormWeightBuffer(qWeight, qWeightBuffer, headDim);
  assertRMSNormWeightBuffer(kWeight, kWeightBuffer, headDim);
  const qWeightDtype = resolveNormWeightDtype(qWeight, headDim);
  const kWeightDtype = resolveNormWeightDtype(kWeight, headDim);

  const paddedHeadDim = padToQ4KBlock(headDim);
  const qOutputSize = qRows * paddedHeadDim * bytesPerElement;
  const kOutputSize = kRows * paddedHeadDim * bytesPerElement;
  let qOutputBuffer = null;
  let kOutputBuffer = null;

  try {
    qOutputBuffer = acquireBuffer(qOutputSize, undefined, 'rmsnorm_qk_q_output');
    kOutputBuffer = acquireBuffer(kOutputSize, undefined, 'rmsnorm_qk_k_output');
    const dispatchPlan = planRMSNormDispatch(target, qRows + kRows);
    await unifiedKernelWrapper(
      'rmsnorm_qk',
      target,
      selectRMSNormQKVariant(qTensor.dtype),
      [qTensor, qWeightBuffer, qOutputBuffer, kTensor, kWeightBuffer, kOutputBuffer],
      {
        q_rows: qRows,
        k_rows: kRows,
        head_dim: headDim,
        row_stride: dispatchPlan.tokenStride,
        eps,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        Q_WEIGHT_IS_F16: qWeightDtype === 'f16',
        K_WEIGHT_IS_F16: kWeightDtype === 'f16',
      },
      null,
      'rmsnorm_qk'
    );

    return {
      q: createTensor(qOutputBuffer, qTensor.dtype, [qRows, headDim], 'rmsnorm_qk_q_output'),
      k: createTensor(kOutputBuffer, kTensor.dtype, [kRows, headDim], 'rmsnorm_qk_k_output'),
    };
  } catch (error) {
    if (qOutputBuffer) {
      releaseBuffer(qOutputBuffer);
    }
    if (kOutputBuffer) {
      releaseBuffer(kOutputBuffer);
    }
    throw error;
  }
}

export async function runRMSNormQK(qTensor, kTensor, qWeight, kWeight, eps, options = {}) {
  return rmsNormQK(null, qTensor, kTensor, qWeight, kWeight, eps, options);
}

export async function recordRMSNormQK(recorder, qTensor, kTensor, qWeight, kWeight, eps, options = {}) {
  return rmsNormQK(recorder, qTensor, kTensor, qWeight, kWeight, eps, options);
}
