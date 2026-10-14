import { createTensor, dtypeBytes } from '../tensor.js';
import { planSplitQKVDispatch } from './split_qkv.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`RoPE QK requires ${label} to be a positive integer.`);
  }
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`RoPE QK requires ${label} to be a non-negative integer.`);
  }
}

function assertRopeGeometry(options) {
  const { headDim, rotaryDim, pairSpanDim } = options;
  assertPositiveInteger(headDim, 'headDim');
  if ((headDim % 2) !== 0) {
    throw new Error(`RoPE QK headDim must be even, got ${headDim}.`);
  }
  assertPositiveInteger(rotaryDim, 'rotaryDim');
  if ((rotaryDim % 2) !== 0) {
    throw new Error(`RoPE QK rotaryDim must be even, got ${rotaryDim}.`);
  }
  if (rotaryDim > headDim) {
    throw new Error(`RoPE QK rotaryDim must be <= headDim; got ${rotaryDim} for headDim ${headDim}.`);
  }
  assertPositiveInteger(pairSpanDim, 'pairSpanDim');
  if ((pairSpanDim % 2) !== 0) {
    throw new Error(`RoPE QK pairSpanDim must be even, got ${pairSpanDim}.`);
  }
  if (pairSpanDim < rotaryDim || pairSpanDim > headDim) {
    throw new Error(
      `RoPE QK pairSpanDim must be in [rotaryDim, headDim]; got ${pairSpanDim} ` +
      `for rotaryDim ${rotaryDim} and headDim ${headDim}.`
    );
  }
}

function bufferOf(bufferOrTensor) {
  return bufferOrTensor?.buffer || bufferOrTensor;
}

function assertBufferCapacity(buffer, requiredBytes, label) {
  if (Number.isFinite(buffer?.size) && requiredBytes > buffer.size) {
    throw new Error(`RoPE QK ${label} buffer is smaller than requested range (${requiredBytes} > ${buffer.size} bytes).`);
  }
}

function selectRoPEQKVariant(dtype, precision) {
  return selectRuleValue('ropeQk', 'variant', { useF16: dtype === 'f16', precision });
}

export function resolveRoPEQKPipeline(dtype, precision) {
  return { operation: 'rope_qk', variant: selectRoPEQKVariant(dtype, precision), constants: null };
}

export function canUseRoPEQK(qTensor, kTensor, options = {}) {
  if (options.reusesSharedKV === true) {
    return false;
  }
  if (!qTensor || !kTensor || qTensor.buffer === kTensor.buffer) {
    return false;
  }
  return qTensor.dtype === kTensor.dtype && (qTensor.dtype === 'f32' || qTensor.dtype === 'f16');
}

async function _ropeQK(target, qTensor, kTensor, freqsCos, freqsSin, seqLen, options = {}) {
  const {
    numQHeads,
    numKVHeads,
    headDim,
    rotaryDim = headDim,
    pairSpanDim = rotaryDim,
    interleaved = false,
    startPos = 0,
  } = options;

  assertNonNegativeInteger(seqLen, 'seqLen');
  assertNonNegativeInteger(startPos, 'startPos');
  assertPositiveInteger(numQHeads, 'numQHeads');
  assertPositiveInteger(numKVHeads, 'numKVHeads');
  assertRopeGeometry({ headDim, rotaryDim, pairSpanDim });
  if (qTensor.dtype !== kTensor.dtype) {
    throw new Error(`RoPE QK requires matching Q/K dtypes, got ${qTensor.dtype} and ${kTensor.dtype}.`);
  }
  if (qTensor.dtype !== 'f32' && qTensor.dtype !== 'f16') {
    throw new Error(`RoPE QK requires f32 or f16 tensors, got ${qTensor.dtype}.`);
  }
  const hasMetadata = value => value.precision !== undefined || value.components !== undefined || value.elementStrideBytes !== undefined;
  const paired = hasMetadata(qTensor) || hasMetadata(kTensor);
  if (paired) {
    for (const tensor of [qTensor, kTensor]) {
      if (tensor.dtype !== 'f32' || tensor.precision !== 'two-f32' || tensor.components !== 2 || tensor.elementStrideBytes !== 8) {
        throw new Error('[PairedRoPE] Matching explicit interleaved two-F32 Q/K required.');
      }
    }
    if (options.frequencyPrecision !== 'two-f32') throw new Error('[PairedRoPE] Paired frequency table identity required.');
    if (qTensor.buffer === kTensor.buffer || [qTensor.buffer, kTensor.buffer].includes(bufferOf(freqsCos))
      || [qTensor.buffer, kTensor.buffer].includes(bufferOf(freqsSin))) throw new Error('[PairedRoPE] Q/K/frequency buffers may not alias.');
  } else if (options.frequencyPrecision === 'two-f32') throw new Error('[RoPE] Scalar inputs cannot consume paired frequencies.');
  const representation = paired ? { precision: 'two-f32', components: 2, elementStrideBytes: 8 } : undefined;
  if (seqLen === 0) {
    return {
      q: createTensor(qTensor.buffer, qTensor.dtype, [...qTensor.shape], 'rope_q_output', representation),
      k: createTensor(kTensor.buffer, kTensor.dtype, [...kTensor.shape], 'rope_k_output', representation),
    };
  }

  const bytesPerElement = paired ? 8 : dtypeBytes(qTensor.dtype);
  const qElementCount = seqLen * numQHeads * headDim;
  const kElementCount = seqLen * numKVHeads * headDim;
  const halfDim = rotaryDim / 2;
  const freqElementCount = (startPos + seqLen) * halfDim;
  assertBufferCapacity(qTensor.buffer, qElementCount * bytesPerElement, 'Q input');
  assertBufferCapacity(kTensor.buffer, kElementCount * bytesPerElement, 'K input');
  assertBufferCapacity(bufferOf(freqsCos), freqElementCount * (paired ? 8 : 4), 'cos frequencies');
  assertBufferCapacity(bufferOf(freqsSin), freqElementCount * (paired ? 8 : 4), 'sin frequencies');

  const pipeline = resolveRoPEQKPipeline(qTensor.dtype, qTensor.precision);
  const totalPairs = seqLen * (numQHeads + numKVHeads) * halfDim;
  const workgroups = paired ? planSplitQKVDispatch(totalPairs) : Math.ceil(totalPairs / WORKGROUP_SIZES.DEFAULT);

  await unifiedKernelWrapper(
    pipeline.operation,
    target,
    pipeline.variant,
    [qTensor, kTensor, freqsCos, freqsSin],
    {
      seq_len: seqLen,
      num_q_heads: numQHeads,
      num_kv_heads: numKVHeads,
      head_dim: headDim,
      start_pos: startPos,
      rotary_dim: rotaryDim,
      interleaved: interleaved ? 1 : 0,
      pair_span_dim: pairSpanDim,
    },
    workgroups,
    pipeline.constants,
    null,
    'rope_qk'
  );

  return {
    q: createTensor(qTensor.buffer, qTensor.dtype, [...qTensor.shape], 'rope_q_output', representation),
    k: createTensor(kTensor.buffer, kTensor.dtype, [...kTensor.shape], 'rope_k_output', representation),
  };
}

export async function runRoPEQK(qTensor, kTensor, freqsCos, freqsSin, seqLen, options = {}) {
  return _ropeQK(null, qTensor, kTensor, freqsCos, freqsSin, seqLen, options);
}

export async function recordRoPEQK(recorder, qTensor, kTensor, freqsCos, freqsSin, seqLen, options = {}) {
  return _ropeQK(recorder, qTensor, kTensor, freqsCos, freqsSin, seqLen, options);
}
