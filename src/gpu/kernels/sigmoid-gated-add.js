import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @type {typeof import('./sigmoid-gated-add.js').validateSigmoidGatedAdd} */
export function validateSigmoidGatedAdd(input, gate, residual, options) {
  const { numTokens, hiddenSize, swigluLimit } = options;
  if (!Number.isSafeInteger(numTokens) || numTokens < 1
    || !Number.isSafeInteger(hiddenSize) || hiddenSize < 1
    || !Number.isSafeInteger(numTokens * hiddenSize) || numTokens * hiddenSize > 0xffffffff) {
    throw new Error('[SigmoidGatedAdd] Explicit positive token/hidden geometry must fit u32.');
  }
  if (swigluLimit !== null && !Number.isFinite(swigluLimit)) {
    throw new Error('[SigmoidGatedAdd] swigluLimit must be null or a finite value.');
  }
  for (const tensor of [input, residual]) {
    if (tensor?.dtype !== 'f32' || tensor.shape?.length !== 2
      || tensor.shape[0] !== numTokens || tensor.shape[1] !== hiddenSize
      || !Number.isSafeInteger(tensor.buffer?.size) || tensor.buffer.size < numTokens * hiddenSize * 4) {
      throw new Error('[SigmoidGatedAdd] Input/residual require explicit F32 token/hidden matrices.');
    }
  }
  if (gate?.dtype !== 'f32' || gate.shape?.length !== 2
    || gate.shape[0] !== numTokens || gate.shape[1] !== 1 || !Number.isSafeInteger(gate.buffer?.size) || gate.buffer.size < numTokens * 4) {
    throw new Error('[SigmoidGatedAdd] Gate requires one F32 scalar per token.');
  }
  if (options.outputBuffer && (!Number.isSafeInteger(options.outputBuffer.size) || options.outputBuffer.size < numTokens * hiddenSize * 4
    || [input.buffer, gate.buffer, residual.buffer].includes(options.outputBuffer))) {
    throw new Error('[SigmoidGatedAdd] Output must have sufficient capacity and distinct ownership.');
  }
}

/** @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {import('../tensor.js').Tensor} gate
 * @param {import('../tensor.js').Tensor} residual
 * @param {import('./sigmoid-gated-add.js').SigmoidGatedAddOptions} options */
async function execute(target, input, gate, residual, options) {
  validateSigmoidGatedAdd(input, gate, residual, options);
  const { numTokens, hiddenSize, swigluLimit, outputBuffer = null } = options;
  const device = target?.device ?? getDevice();
  const maxGroups = device?.limits?.maxComputeWorkgroupsPerDimension;
  if (typeof maxGroups !== 'number' || !Number.isSafeInteger(maxGroups) || maxGroups < 1) {
    throw new Error('[SigmoidGatedAdd] Active device dispatch limits are required.');
  }
  const size = numTokens * hiddenSize;
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const stride = Math.min(size, maxGroups * workgroupSize);
  const groups = [Math.ceil(stride / workgroupSize), Math.ceil(size / stride), 1];
  if (groups[1] > maxGroups) throw new Error('[SigmoidGatedAdd] Geometry exceeds device dispatch capacity.');
  return withKernelOutput(target, outputBuffer, size * 4, 'sigmoid_gated_add_output', async (output) => {
    await unifiedKernelWrapper('sigmoid_gated_add', target, 'f32', [input, gate, residual, output],
      { size, dispatch_stride: stride, clamp_max: swigluLimit ?? 0, _pad0: 0 }, groups,
      { HIDDEN_SIZE: hiddenSize, WORKGROUP_SIZE: workgroupSize });
    return createTensor(output, 'f32', [numTokens, hiddenSize], 'sigmoid_gated_add_output');
  });
}

/** @type {typeof import('./sigmoid-gated-add.js').runSigmoidGatedAdd} */
export function runSigmoidGatedAdd(input, gate, residual, options) {
  return execute(null, input, gate, residual, options);
}

/** @type {typeof import('./sigmoid-gated-add.js').recordSigmoidGatedAdd} */
export function recordSigmoidGatedAdd(recorder, input, gate, residual, options) {
  return execute(recorder, input, gate, residual, options);
}
