import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const PAIRED = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });

/** @param {import('../tensor.js').Tensor} input */
function logicalCount(input) {
  if (input?.dtype !== 'f32' || !Array.isArray(input.shape) || input.shape.length === 0
    || input.shape.some(size => !Number.isSafeInteger(size) || size <= 0)) {
    throw new Error('[PairedCast] A positive, explicitly shaped F32 tensor is required.');
  }
  const count = input.shape.reduce((total, size) => total * size, 1);
  if (!Number.isSafeInteger(count) || count > 0xffffffff) {
    throw new Error('[PairedCast] Logical element count exceeds the U32 shader contract.');
  }
  return count;
}

/** @type {typeof import('./paired-cast.js').resolvePairedCastPipeline} */
export function resolvePairedCastPipeline(pairedOutput) {
  return { operation: 'cast', variant: pairedOutput ? 'f32_to_two_f32' : 'two_f32_to_f16', constants: null };
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./types.js').OutputBufferOptions} options
 * @param {boolean} pairedOutput
 */
async function execute(target, input, options, pairedOutput) {
  const pipeline = resolvePairedCastPipeline(pairedOutput);
  const count = logicalCount(input);
  const pairedInput = input.precision === 'two-f32';
  if (pairedOutput && (input.precision !== undefined || input.components !== undefined
    || input.elementStrideBytes !== undefined)) {
    throw new Error('[PairedCast] Expansion requires a scalar F32 input.');
  }
  if (!pairedOutput && (!pairedInput || input.components !== 2 || input.elementStrideBytes !== 8)) {
    throw new Error('[PairedCast] Direct F16 conversion requires explicit interleaved two-F32 input.');
  }
  const inputBytes = count * (pairedInput ? 8 : 4);
  if (!input.buffer || input.buffer.size < inputBytes) throw new Error('[PairedCast] Input buffer is undersized.');
  const outputBytes = pairedOutput ? count * 8 : Math.ceil(count / 2) * 4;
  const supplied = options.outputBuffer;
  if (supplied && (supplied === input.buffer || supplied.size < outputBytes)) {
    throw new Error('[PairedCast] Output must be a distinct, sufficiently sized buffer.');
  }
  const output = supplied ?? acquireBuffer(outputBytes, undefined, `cast_${pipeline.variant}`);
  const dispatchCount = pairedOutput ? count : Math.ceil(count / 2);
  const workgroups = Math.ceil(dispatchCount / WORKGROUP_SIZES.DEFAULT);
  const x = Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS);
  try {
    await unifiedKernelWrapper(pipeline.operation, target, pipeline.variant, [input, output], { size: count },
      [x, Math.ceil(workgroups / x), 1], pipeline.constants);
    return pairedOutput
      ? createTensor(output, 'f32', [...input.shape], 'paired_activation', PAIRED)
      : createTensor(output, 'f16', [...input.shape], 'paired_direct_half');
  } catch (error) {
    if (!supplied) releaseBuffer(output);
    throw error;
  }
}

/** @type {typeof import('./paired-cast.js').expandF32ToPaired} */
export function expandF32ToPaired(input, options = {}) {
  return execute(null, input, options, true);
}
/** @type {typeof import('./paired-cast.js').recordExpandF32ToPaired} */
export function recordExpandF32ToPaired(recorder, input, options = {}) {
  return execute(recorder, input, options, true);
}
/** @type {typeof import('./paired-cast.js').castPairedToF16} */
export function castPairedToF16(input, options = {}) {
  return execute(null, input, options, false);
}
/** @type {typeof import('./paired-cast.js').recordCastPairedToF16} */
export function recordCastPairedToF16(recorder, input, options = {}) {
  return execute(recorder, input, options, false);
}
