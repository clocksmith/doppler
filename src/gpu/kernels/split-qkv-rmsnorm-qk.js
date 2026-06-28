import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import {
  assertRMSNormWeightBuffer,
  planRMSNormDispatch,
  resolveNormWeightDtype,
} from './rmsnorm.js';
import { WORKGROUP_SIZES } from './constants.js';

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`split_qkv_rmsnorm_qk requires ${label} to be a positive integer.`);
  }
}

function assertQKVSizes(options) {
  const {
    qSize,
    kSize,
    vSize,
    numHeads,
    numKVHeads,
    headDim,
  } = options;
  if (qSize !== numHeads * headDim) {
    throw new Error(`split_qkv_rmsnorm_qk qSize mismatch: got ${qSize}, expected ${numHeads * headDim}.`);
  }
  if (kSize !== numKVHeads * headDim) {
    throw new Error(`split_qkv_rmsnorm_qk kSize mismatch: got ${kSize}, expected ${numKVHeads * headDim}.`);
  }
  if (vSize !== numKVHeads * headDim) {
    throw new Error(`split_qkv_rmsnorm_qk vSize mismatch: got ${vSize}, expected ${numKVHeads * headDim}.`);
  }
}

export function canUseSplitQKVRMSNormQK(qkvTensor, options = {}) {
  if (!qkvTensor || qkvTensor.dtype !== 'f32') {
    return false;
  }
  if (options.skipKNorm === true || options.allowUnitQKNorm === true) {
    return false;
  }
  return true;
}

async function splitQKVRMSNormQK(target, qkvTensor, qWeight, kWeight, eps, options = {}) {
  const {
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
    qSize,
    kSize,
    vSize,
    rmsNormWeightOffset = false,
  } = options;

  assertPositiveInteger(numTokens, 'numTokens');
  assertPositiveInteger(numHeads, 'numHeads');
  assertPositiveInteger(numKVHeads, 'numKVHeads');
  assertPositiveInteger(headDim, 'headDim');
  assertPositiveInteger(qSize, 'qSize');
  assertPositiveInteger(kSize, 'kSize');
  assertPositiveInteger(vSize, 'vSize');
  assertQKVSizes(options);
  if (qkvTensor.dtype !== 'f32') {
    throw new Error(`split_qkv_rmsnorm_qk requires f32 QKV input, got ${qkvTensor.dtype}.`);
  }

  const qWeightBuffer = getBuffer(qWeight);
  const kWeightBuffer = getBuffer(kWeight);
  assertRMSNormWeightBuffer(qWeight, qWeightBuffer, headDim);
  assertRMSNormWeightBuffer(kWeight, kWeightBuffer, headDim);
  const qWeightDtype = resolveNormWeightDtype(qWeight, headDim);
  const kWeightDtype = resolveNormWeightDtype(kWeight, headDim);

  const qRows = numTokens * numHeads;
  const kRows = numTokens * numKVHeads;
  const qkRows = qRows + kRows;
  const totalV = numTokens * vSize;
  const vWorkgroups = Math.ceil(totalV / WORKGROUP_SIZES.DEFAULT);
  const totalWorkgroups = qkRows + vWorkgroups;
  const dispatchPlan = planRMSNormDispatch(target, totalWorkgroups);
  const bytesPerElement = dtypeBytes(qkvTensor.dtype);
  const paddedHeadDim = padToQ4KBlock(headDim);
  const qOutputSize = qRows * paddedHeadDim * bytesPerElement;
  const kOutputSize = kRows * paddedHeadDim * bytesPerElement;
  const vOutputSize = totalV * bytesPerElement;
  let qOutputBuffer = null;
  let kOutputBuffer = null;
  let vOutputBuffer = null;

  try {
    qOutputBuffer = acquireBuffer(qOutputSize, undefined, 'split_qkv_rmsnorm_qk_q_output');
    kOutputBuffer = acquireBuffer(kOutputSize, undefined, 'split_qkv_rmsnorm_qk_k_output');
    vOutputBuffer = acquireBuffer(vOutputSize, undefined, 'split_qkv_rmsnorm_qk_v_output');
    await unifiedKernelWrapper(
      'split_qkv_rmsnorm_qk',
      target,
      'default',
      [qkvTensor, qWeightBuffer, kWeightBuffer, qOutputBuffer, kOutputBuffer, vOutputBuffer],
      {
        num_tokens: numTokens,
        q_size: qSize,
        k_size: kSize,
        v_size: vSize,
        num_heads: numHeads,
        num_kv_heads: numKVHeads,
        head_dim: headDim,
        workgroup_stride: dispatchPlan.tokenStride,
        qk_rows: qkRows,
        total_v: totalV,
        eps,
        _pad0: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        Q_WEIGHT_IS_F16: qWeightDtype === 'f16',
        K_WEIGHT_IS_F16: kWeightDtype === 'f16',
      },
      null,
      'split_qkv_rmsnorm_qk'
    );

    return {
      Q: createTensor(qOutputBuffer, 'f32', [qRows, headDim], 'split_qkv_rmsnorm_qk_q_output'),
      K: createTensor(kOutputBuffer, 'f32', [kRows, headDim], 'split_qkv_rmsnorm_qk_k_output'),
      V: createTensor(vOutputBuffer, 'f32', [numTokens, vSize], 'split_qkv_rmsnorm_qk_v_output'),
    };
  } catch (error) {
    if (qOutputBuffer) {
      releaseBuffer(qOutputBuffer);
    }
    if (kOutputBuffer) {
      releaseBuffer(kOutputBuffer);
    }
    if (vOutputBuffer) {
      releaseBuffer(vOutputBuffer);
    }
    throw error;
  }
}

export async function runSplitQKVRMSNormQK(qkvTensor, qWeight, kWeight, eps, options = {}) {
  return splitQKVRMSNormQK(null, qkvTensor, qWeight, kWeight, eps, options);
}

export async function recordSplitQKVRMSNormQK(recorder, qkvTensor, qWeight, kWeight, eps, options = {}) {
  return splitQKVRMSNormQK(recorder, qkvTensor, qWeight, kWeight, eps, options);
}
