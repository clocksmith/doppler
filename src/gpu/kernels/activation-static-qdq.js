import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';

function resolveCount(tensor, countOverride) {
  if (Number.isFinite(countOverride) && countOverride > 0) {
    return Math.floor(countOverride);
  }
  if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
    return tensor.shape.reduce((acc, value) => acc * value, 1);
  }
  return Math.floor(tensor.buffer.size / dtypeBytes(tensor.dtype));
}

async function executeActivationStaticQdq(target, input, scale, options = {}) {
  if (input.dtype !== 'f32') {
    throw new Error(`activation_static_qdq: unsupported dtype ${input.dtype}.`);
  }

  const { count, outputBuffer = null, qmin = -128, qmax = 127 } = options;
  const resolvedScale = Number(scale);
  const resolvedQmin = Number(qmin);
  const resolvedQmax = Number(qmax);
  if (!Number.isFinite(resolvedScale) || resolvedScale <= 0) {
    throw new Error(`activation_static_qdq: scale must be > 0. Got ${String(scale)}.`);
  }
  if (!Number.isFinite(resolvedQmin) || !Number.isFinite(resolvedQmax) || resolvedQmin > resolvedQmax) {
    throw new Error(
      `activation_static_qdq: invalid qmin/qmax range (${String(qmin)}, ${String(qmax)}).`
    );
  }

  const inferredCount = resolveCount(input, count);
  const outputSize = inferredCount * dtypeBytes(input.dtype);
  const ownsOutput = outputBuffer == null;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'activation_static_qdq_output');

  try {
    await unifiedKernelWrapper(
      'activation_static_qdq',
      target,
      'default',
      [input, output],
      {
        size: inferredCount,
        scale: resolvedScale,
        invScale: 1 / resolvedScale,
        qmin: resolvedQmin,
        qmax: resolvedQmax,
      },
      Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT)
    );
    return createTensor(output, input.dtype, [...input.shape], 'activation_static_qdq_output');
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(output);
    }
    throw error;
  }
}

export async function runActivationStaticQdq(input, scale, options = {}) {
  return executeActivationStaticQdq(null, input, scale, options);
}

export async function recordActivationStaticQdq(recorder, input, scale, options = {}) {
  return executeActivationStaticQdq(recorder, input, scale, options);
}
