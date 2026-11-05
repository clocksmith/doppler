import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

// qwen4_exp hyper-connection operators (Qwen4ExpTextGatedResidual and the
// decoder-layer injection). Each operator is independently dispatchable; the
// low-rank projections between them are ordinary matmuls owned by callers.

const WORKGROUP_SIZE = 256;
const MAX_U32 = 0xffffffff;
/** @type {Readonly<Record<string, 'low_rank_silu' | 'mix_sigmoid' | 'inject_gate'>>} */
const ACTIVATION_VARIANTS = Object.freeze({
  low_rank_silu: 'low_rank_silu',
  mix_sigmoid: 'mix_sigmoid',
  inject_gate: 'inject_gate',
});

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[HyperConnection] ${message}`);
}

/** @type {typeof import('./hyper-connection.js').validateHyperConnectionGeometry} */
export function validateHyperConnectionGeometry({ numTokens, streamCount, streamWidth }) {
  for (const [name, value] of /** @type {[string, number][]} */ ([['numTokens', numTokens], ['streamCount', streamCount], ['streamWidth', streamWidth]])) {
    if (!Number.isSafeInteger(value) || value < 1) fail(`${name} must be a positive integer; got ${String(value)}.`);
  }
  if (streamCount < 2) fail('streamCount must exceed 1.');
  const elements = numTokens * streamCount * streamWidth;
  if (!Number.isSafeInteger(elements) || elements > MAX_U32) fail('stream geometry must fit u32 indexing.');
  return elements;
}

/**
 * @param {import('../tensor.js').Tensor} tensor
 * @param {readonly number[]} shape
 * @param {string} label
 */
function assertF32Tensor(tensor, shape, label) {
  const elements = shape.reduce((product, dim) => product * dim, 1);
  if (tensor?.dtype !== 'f32' || !Array.isArray(tensor.shape) || tensor.shape.length !== shape.length
    || tensor.shape.some((dim, index) => dim !== shape[index])) {
    fail(`${label} must be an f32 tensor of shape [${shape.join(', ')}]; got ${tensor?.dtype} [${tensor?.shape?.join(', ')}].`);
  }
  if (!Number.isSafeInteger(tensor.buffer?.size) || tensor.buffer.size < elements * 4) {
    fail(`${label} buffer holds fewer than ${elements} f32 values.`);
  }
}

/**
 * @param {GPUBuffer | null} outputBuffer
 * @param {number} bytes
 * @param {readonly (import('../tensor.js').Tensor)[]} inputs
 */
function assertOutputBuffer(outputBuffer, bytes, inputs) {
  if (!outputBuffer) return;
  if (!Number.isSafeInteger(outputBuffer.size) || outputBuffer.size < bytes) fail('outputBuffer is too small.');
  if (inputs.some((tensor) => tensor.buffer === outputBuffer)) fail('outputBuffer must not alias an input.');
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {number} units
 * @param {number} unitsPerGroup
 */
function dispatchGrid(target, units, unitsPerGroup) {
  const device = target?.device ?? getDevice();
  const maxGroups = device?.limits?.maxComputeWorkgroupsPerDimension;
  if (typeof maxGroups !== 'number' || !Number.isSafeInteger(maxGroups) || maxGroups < 1) {
    fail('active device dispatch limits are required.');
  }
  const groups = Math.ceil(units / unitsPerGroup);
  const columns = Math.min(groups, maxGroups);
  const rows = Math.ceil(groups / columns);
  if (rows > maxGroups) fail('geometry exceeds device dispatch capacity.');
  return { columns, grid: [columns, rows, 1] };
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {import('../tensor.js').Tensor} weight
 * @param {import('./hyper-connection.js').HyperConnectionNormOptions} options
 */
async function executeNorm(target, input, weight, options) {
  const { numTokens, streamCount, streamWidth, eps, outputBuffer = null } = options;
  const elements = validateHyperConnectionGeometry({ numTokens, streamCount, streamWidth });
  if (typeof eps !== 'number' || !Number.isFinite(eps) || eps <= 0) fail('eps must be a positive finite number.');
  assertF32Tensor(input, [numTokens, streamCount * streamWidth], 'norm input');
  assertF32Tensor(weight, [streamCount * streamWidth], 'norm weight');
  assertOutputBuffer(outputBuffer, elements * 4, [input, weight]);
  const groups = numTokens * streamCount;
  const { columns, grid } = dispatchGrid(target, groups, 1);
  return withKernelOutput(target, outputBuffer, elements * 4, 'hyper_connection_norm_output', async (output) => {
    await unifiedKernelWrapper('hyper_connection_norm', target, 'f32', [input, weight, output],
      { group_count: groups, group_stride: columns, eps, _pad0: 0 }, grid,
      { WORKGROUP_SIZE, STREAM_COUNT: streamCount, STREAM_WIDTH: streamWidth });
    return createTensor(output, 'f32', [numTokens, streamCount * streamWidth], 'hyper_connection_norm_output');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./hyper-connection.js').HyperConnectionActivationOptions} options
 */
async function executeActivation(target, input, options) {
  const { mode, rows, columns, streamCount, outputBuffer = null } = options;
  const variant = ACTIVATION_VARIANTS[mode];
  if (!variant) fail(`unknown activation mode ${JSON.stringify(mode)}.`);
  if (!Number.isSafeInteger(streamCount) || streamCount < 2) fail('streamCount must exceed 1.');
  for (const [name, value] of /** @type {[string, number][]} */ ([['rows', rows], ['columns', columns]])) {
    if (!Number.isSafeInteger(value) || value < 1) fail(`activation ${name} must be a positive integer.`);
  }
  const size = rows * columns;
  if (size > MAX_U32) fail('activation size must fit u32 indexing.');
  assertF32Tensor(input, [rows, columns], `${mode} input`);
  assertOutputBuffer(outputBuffer, size * 4, [input]);
  const { columns: stride, grid } = dispatchGrid(target, size, WORKGROUP_SIZE);
  return withKernelOutput(target, outputBuffer, size * 4, `hyper_connection_${mode}_output`, async (output) => {
    await unifiedKernelWrapper('hyper_connection_activation', target, variant, [input, output],
      { size, dispatch_stride: stride * WORKGROUP_SIZE, _pad0: 0, _pad1: 0 }, grid,
      { WORKGROUP_SIZE, STREAM_COUNT: streamCount });
    return createTensor(output, 'f32', [rows, columns], `hyper_connection_${mode}_output`);
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} mix
 * @param {import('../tensor.js').Tensor} normed
 * @param {import('./hyper-connection.js').HyperConnectionStreamOptions} options
 */
async function executeBranch(target, mix, normed, options) {
  const { numTokens, streamCount, streamWidth, outputBuffer = null } = options;
  validateHyperConnectionGeometry({ numTokens, streamCount, streamWidth });
  assertF32Tensor(mix, [numTokens, streamCount * streamWidth], 'branch mix');
  assertF32Tensor(normed, [numTokens, streamCount * streamWidth], 'branch normed streams');
  const size = numTokens * streamWidth;
  assertOutputBuffer(outputBuffer, size * 4, [mix, normed]);
  const { columns: stride, grid } = dispatchGrid(target, size, WORKGROUP_SIZE);
  return withKernelOutput(target, outputBuffer, size * 4, 'hyper_connection_branch_output', async (output) => {
    await unifiedKernelWrapper('hyper_connection_branch', target, 'f32', [mix, normed, output],
      { size, dispatch_stride: stride * WORKGROUP_SIZE, _pad0: 0, _pad1: 0 }, grid,
      { WORKGROUP_SIZE, STREAM_COUNT: streamCount, STREAM_WIDTH: streamWidth });
    return createTensor(output, 'f32', [numTokens, streamWidth], 'hyper_connection_branch_output');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} streams
 * @param {import('../tensor.js').Tensor} branch
 * @param {import('../tensor.js').Tensor} gate
 * @param {import('./hyper-connection.js').HyperConnectionStreamOptions} options
 */
async function executeInject(target, streams, branch, gate, options) {
  const { numTokens, streamCount, streamWidth, outputBuffer = null } = options;
  const size = validateHyperConnectionGeometry({ numTokens, streamCount, streamWidth });
  assertF32Tensor(streams, [numTokens, streamCount * streamWidth], 'inject streams');
  assertF32Tensor(branch, [numTokens, streamWidth], 'inject branch output');
  assertF32Tensor(gate, [numTokens, streamCount], 'inject gate');
  assertOutputBuffer(outputBuffer, size * 4, [streams, branch, gate]);
  const { columns: stride, grid } = dispatchGrid(target, size, WORKGROUP_SIZE);
  return withKernelOutput(target, outputBuffer, size * 4, 'hyper_connection_inject_output', async (output) => {
    await unifiedKernelWrapper('hyper_connection_inject', target, 'f32', [streams, branch, gate, output],
      { size, dispatch_stride: stride * WORKGROUP_SIZE, _pad0: 0, _pad1: 0 }, grid,
      { WORKGROUP_SIZE, STREAM_COUNT: streamCount, STREAM_WIDTH: streamWidth });
    return createTensor(output, 'f32', [numTokens, streamCount * streamWidth], 'hyper_connection_inject_output');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} embeds
 * @param {import('./hyper-connection.js').HyperConnectionStreamOptions} options
 */
async function executeReplicate(target, embeds, options) {
  const { numTokens, streamCount, streamWidth, outputBuffer = null } = options;
  const size = validateHyperConnectionGeometry({ numTokens, streamCount, streamWidth });
  assertF32Tensor(embeds, [numTokens, streamWidth], 'replicate embeddings');
  assertOutputBuffer(outputBuffer, size * 4, [embeds]);
  const { columns: stride, grid } = dispatchGrid(target, size, WORKGROUP_SIZE);
  return withKernelOutput(target, outputBuffer, size * 4, 'hyper_connection_streams', async (output) => {
    await unifiedKernelWrapper('hyper_connection_activation', target, 'replicate', [embeds, output],
      { size, dispatch_stride: stride * WORKGROUP_SIZE, _pad0: 0, _pad1: 0 }, grid,
      { WORKGROUP_SIZE, STREAM_COUNT: streamCount, STREAM_WIDTH: streamWidth });
    return createTensor(output, 'f32', [numTokens, streamCount * streamWidth], 'hyper_connection_streams');
  });
}

/** @type {typeof import('./hyper-connection.js').runHyperConnectionReplicate} */
export function runHyperConnectionReplicate(embeds, options) {
  return executeReplicate(null, embeds, options);
}

/** @type {typeof import('./hyper-connection.js').recordHyperConnectionReplicate} */
export function recordHyperConnectionReplicate(recorder, embeds, options) {
  return executeReplicate(recorder, embeds, options);
}

/** @type {typeof import('./hyper-connection.js').runHyperConnectionNorm} */
export function runHyperConnectionNorm(input, weight, options) {
  return executeNorm(null, input, weight, options);
}

/** @type {typeof import('./hyper-connection.js').recordHyperConnectionNorm} */
export function recordHyperConnectionNorm(recorder, input, weight, options) {
  return executeNorm(recorder, input, weight, options);
}

/** @type {typeof import('./hyper-connection.js').runHyperConnectionActivation} */
export function runHyperConnectionActivation(input, options) {
  return executeActivation(null, input, options);
}

/** @type {typeof import('./hyper-connection.js').recordHyperConnectionActivation} */
export function recordHyperConnectionActivation(recorder, input, options) {
  return executeActivation(recorder, input, options);
}

/** @type {typeof import('./hyper-connection.js').runHyperConnectionBranch} */
export function runHyperConnectionBranch(mix, normed, options) {
  return executeBranch(null, mix, normed, options);
}

/** @type {typeof import('./hyper-connection.js').recordHyperConnectionBranch} */
export function recordHyperConnectionBranch(recorder, mix, normed, options) {
  return executeBranch(recorder, mix, normed, options);
}

/** @type {typeof import('./hyper-connection.js').runHyperConnectionInject} */
export function runHyperConnectionInject(streams, branch, gate, options) {
  return executeInject(null, streams, branch, gate, options);
}

/** @type {typeof import('./hyper-connection.js').recordHyperConnectionInject} */
export function recordHyperConnectionInject(recorder, streams, branch, gate, options) {
  return executeInject(recorder, streams, branch, gate, options);
}
