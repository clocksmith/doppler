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
    throw new Error(`split_qkv_rmsnorm_rope_qk requires ${label} to be a positive integer.`);
  }
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`split_qkv_rmsnorm_rope_qk requires ${label} to be a non-negative integer.`);
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
    throw new Error(`split_qkv_rmsnorm_rope_qk qSize mismatch: got ${qSize}, expected ${numHeads * headDim}.`);
  }
  if (kSize !== numKVHeads * headDim) {
    throw new Error(`split_qkv_rmsnorm_rope_qk kSize mismatch: got ${kSize}, expected ${numKVHeads * headDim}.`);
  }
  if (vSize !== numKVHeads * headDim) {
    throw new Error(`split_qkv_rmsnorm_rope_qk vSize mismatch: got ${vSize}, expected ${numKVHeads * headDim}.`);
  }
}

function bufferOf(bufferOrTensor) {
  return bufferOrTensor?.buffer || bufferOrTensor;
}

function assertBufferCapacity(buffer, requiredBytes, label) {
  if (Number.isFinite(buffer?.size) && requiredBytes > buffer.size) {
    throw new Error(
      `split_qkv_rmsnorm_rope_qk ${label} buffer is smaller than requested range ` +
      `(${requiredBytes} > ${buffer.size} bytes).`
    );
  }
}

function supportsFullHeadNonInterleavedRoPE(options) {
  const {
    headDim,
    rotaryDim = headDim,
    pairSpanDim = rotaryDim,
    interleaved = false,
  } = options;
  return Number.isInteger(headDim)
    && headDim > 0
    && (headDim % 2) === 0
    && rotaryDim === headDim
    && pairSpanDim === headDim
    && interleaved === false;
}

function assertSupportedRoPE(options) {
  const {
    headDim,
    rotaryDim = headDim,
    pairSpanDim = rotaryDim,
    interleaved = false,
  } = options;
  if (!supportsFullHeadNonInterleavedRoPE({ headDim, rotaryDim, pairSpanDim, interleaved })) {
    throw new Error(
      'split_qkv_rmsnorm_rope_qk supports only full-head non-interleaved RoPE ' +
      `(headDim=${headDim}, rotaryDim=${rotaryDim}, pairSpanDim=${pairSpanDim}, interleaved=${interleaved}).`
    );
  }
}

export function canUseSplitQKVRMSNormRoPEQK(qkvTensor, options = {}) {
  if (!qkvTensor || qkvTensor.dtype !== 'f32') {
    return false;
  }
  if (options.reusesSharedKV === true || options.skipKNorm === true || options.allowUnitQKNorm === true) {
    return false;
  }
  return supportsFullHeadNonInterleavedRoPE(options);
}

async function splitQKVRMSNormRoPEQK(target, qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options = {}) {
  const {
    numTokens,
    numHeads,
    numKVHeads,
    headDim,
    qSize,
    kSize,
    vSize,
    startPos = 0,
    rmsNormWeightOffset = false,
    f16KVCacheWrite = null,
  } = options;

  assertPositiveInteger(numTokens, 'numTokens');
  assertPositiveInteger(numHeads, 'numHeads');
  assertPositiveInteger(numKVHeads, 'numKVHeads');
  assertPositiveInteger(headDim, 'headDim');
  assertPositiveInteger(qSize, 'qSize');
  assertPositiveInteger(kSize, 'kSize');
  assertPositiveInteger(vSize, 'vSize');
  assertNonNegativeInteger(startPos, 'startPos');
  assertQKVSizes(options);
  assertSupportedRoPE(options);
  if (qkvTensor.dtype !== 'f32') {
    throw new Error(`split_qkv_rmsnorm_rope_qk requires f32 QKV input, got ${qkvTensor.dtype}.`);
  }

  const qWeightBuffer = getBuffer(qWeight);
  const kWeightBuffer = getBuffer(kWeight);
  const cosBuffer = bufferOf(freqsCos);
  const sinBuffer = bufferOf(freqsSin);
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
  const halfDim = headDim / 2;
  const freqElementCount = (startPos + numTokens) * halfDim;
  const writesF16KVCache = f16KVCacheWrite != null;
  assertBufferCapacity(qkvTensor.buffer, numTokens * (qSize + kSize + vSize) * bytesPerElement, 'QKV input');
  assertBufferCapacity(cosBuffer, freqElementCount * 4, 'cos frequencies');
  assertBufferCapacity(sinBuffer, freqElementCount * 4, 'sin frequencies');
  if (writesF16KVCache) {
    assertNonNegativeInteger(f16KVCacheWrite.dstOffset, 'f16KVCacheWrite.dstOffset');
    const requiredCacheElements = f16KVCacheWrite.dstOffset + (numTokens * numKVHeads * headDim);
    assertBufferCapacity(f16KVCacheWrite.keysBuffer, requiredCacheElements * 2, 'f16 KV keys cache');
    assertBufferCapacity(f16KVCacheWrite.valuesBuffer, requiredCacheElements * 2, 'f16 KV values cache');
  }
  let qOutputBuffer = null;
  let kOutputBuffer = null;
  let vOutputBuffer = null;

  try {
    qOutputBuffer = acquireBuffer(qOutputSize, undefined, 'split_qkv_rmsnorm_rope_qk_q_output');
    if (!writesF16KVCache) {
      kOutputBuffer = acquireBuffer(kOutputSize, undefined, 'split_qkv_rmsnorm_rope_qk_k_output');
      vOutputBuffer = acquireBuffer(vOutputSize, undefined, 'split_qkv_rmsnorm_rope_qk_v_output');
    }
    const opName = writesF16KVCache
      ? 'split_qkv_rmsnorm_rope_qk_f16kv_cache'
      : 'split_qkv_rmsnorm_rope_qk';
    const bindings = writesF16KVCache
      ? [
        qkvTensor,
        qWeightBuffer,
        kWeightBuffer,
        cosBuffer,
        sinBuffer,
        qOutputBuffer,
        f16KVCacheWrite.keysBuffer,
        f16KVCacheWrite.valuesBuffer,
      ]
      : [qkvTensor, qWeightBuffer, kWeightBuffer, cosBuffer, sinBuffer, qOutputBuffer, kOutputBuffer, vOutputBuffer];
    await unifiedKernelWrapper(
      opName,
      target,
      'default',
      bindings,
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
        start_pos: startPos,
        half_dim: halfDim,
        eps,
        kv_dst_offset: writesF16KVCache ? f16KVCacheWrite.dstOffset : 0,
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
      writesF16KVCache
        ? 'split_qkv_rmsnorm_rope_qk:f16kv_cache'
        : 'split_qkv_rmsnorm_rope_qk'
    );

    return {
      Q: createTensor(qOutputBuffer, 'f32', [qRows, headDim], 'split_qkv_rmsnorm_rope_qk_q_output'),
      K: writesF16KVCache ? null : createTensor(kOutputBuffer, 'f32', [kRows, headDim], 'split_qkv_rmsnorm_rope_qk_k_output'),
      V: writesF16KVCache ? null : createTensor(vOutputBuffer, 'f32', [numTokens, vSize], 'split_qkv_rmsnorm_rope_qk_v_output'),
      wroteF16KVCache: writesF16KVCache,
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

export async function runSplitQKVRMSNormRoPEQK(qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options = {}) {
  return splitQKVRMSNormRoPEQK(null, qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options);
}

export async function recordSplitQKVRMSNormRoPEQK(recorder, qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options = {}) {
  return splitQKVRMSNormRoPEQK(recorder, qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options);
}
