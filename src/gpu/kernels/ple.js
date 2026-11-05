import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

// qwen4_exp PLE operators: BF16 row decode, the signed-square-root sigmoid
// gate, and the dilated depthwise short convolution with owned state and the
// residual into the streams. Projections and grouped norms are separate ops.

const WORKGROUP_SIZE = 256;
const MAX_X_GROUPS = 65535;

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[PLE] ${message}`);
}

/** @param {unknown} value @param {string} name @param {number} [minimum] */
function integer(value, name, minimum = 1) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) fail(`${name} must be an integer >= ${minimum}.`);
  return value;
}

/** @param {import('../tensor.js').Tensor} tensor @param {readonly number[]} shape @param {string} label */
function assertF32(tensor, shape, label) {
  const elements = shape.reduce((product, dim) => product * dim, 1);
  if (tensor?.dtype !== 'f32' || tensor.shape?.length !== shape.length || tensor.shape.some((dim, i) => dim !== shape[i])) {
    fail(`${label} must be an f32 tensor of shape [${shape.join(', ')}].`);
  }
  if (!Number.isSafeInteger(tensor.buffer?.size) || tensor.buffer.size < elements * 4) fail(`${label} buffer is too small.`);
}

/** @param {import('../command-recorder.js').CommandRecorder | null} target @param {GPUBuffer} buffer */
function releaseFor(target, buffer) {
  if (target && typeof target.beginComputePass === 'function') target.trackTemporaryBuffer(buffer);
  else releaseBuffer(buffer);
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {GPUBuffer} rows
 * @param {import('./ple.js').PleRowDecodeOptions} options
 */
async function executeDecode(target, rows, { numTokens, embedDim }) {
  integer(numTokens, 'numTokens');
  integer(embedDim, 'embedDim', 2);
  if (embedDim % 2 !== 0) fail('embedDim must be even for packed BF16 decoding.');
  const elements = numTokens * embedDim;
  if (!Number.isSafeInteger(rows?.size) || rows.size < elements * 2) fail('row bytes buffer is too small.');
  const pairs = elements / 2;
  const groups = Math.ceil(pairs / WORKGROUP_SIZE);
  const grid = groups > MAX_X_GROUPS ? [MAX_X_GROUPS, Math.ceil(groups / MAX_X_GROUPS), 1] : [groups, 1, 1];
  return withKernelOutput(target, null, elements * 4, 'ple_embedding', async (output) => {
    await unifiedKernelWrapper('bf16_to_f32', target, 'default', [rows, output],
      { num_elements: elements, input_offset: 0, output_offset: 0, _pad: 0 }, grid, { WORKGROUP_SIZE });
    return createTensor(output, 'f32', [numTokens, embedDim], 'ple_embedding');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} keyNormed
 * @param {import('../tensor.js').Tensor} queryNormed
 * @param {import('../tensor.js').Tensor} value
 * @param {import('./ple.js').PleGateOptions} options
 */
async function executeGate(target, keyNormed, queryNormed, value, { numTokens, streamCount, streamWidth }) {
  integer(numTokens, 'numTokens');
  integer(streamCount, 'streamCount', 2);
  integer(streamWidth, 'streamWidth');
  const width = streamCount * streamWidth;
  assertF32(keyNormed, [numTokens, width], 'normalized key');
  assertF32(queryNormed, [numTokens, width], 'normalized query');
  assertF32(value, [numTokens, streamWidth], 'value');
  const device = target?.device ?? getDevice();
  const maxGroups = device?.limits?.maxComputeWorkgroupsPerDimension ?? MAX_X_GROUPS;
  const groups = numTokens * streamCount;
  const columns = Math.min(groups, maxGroups);
  /** @type {GPUBuffer | null} */
  let gateInput = null;
  try {
    gateInput = acquireBuffer(groups * 4, undefined, 'ple_gate_input');
    const capturedGate = gateInput;
    const gated = await withKernelOutput(target, null, numTokens * width * 4, 'ple_gated', async (output) => {
      await unifiedKernelWrapper('ple_gate', target, 'f32', [keyNormed, queryNormed, value, output, capturedGate],
        { num_tokens: numTokens, group_stride: columns, divisor: Math.fround(Math.sqrt(streamWidth)), clamp_min: Math.fround(1e-6) },
        [columns, Math.ceil(groups / columns), 1], { WORKGROUP_SIZE, STREAM_COUNT: streamCount, STREAM_WIDTH: streamWidth });
      return createTensor(output, 'f32', [numTokens, width], 'ple_gated');
    });
    return { gated, gateInput: createTensor(capturedGate, 'f32', [numTokens, streamCount], 'ple_gate_input') };
  } catch (error) {
    if (gateInput) releaseFor(target, gateInput);
    throw error;
  }
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} normed
 * @param {import('../tensor.js').Tensor} gated
 * @param {import('../tensor.js').Tensor} streams
 * @param {import('../tensor.js').Tensor} convWeight
 * @param {GPUBuffer} state
 * @param {import('./ple.js').PleConvOptions} options
 */
async function executeConv(target, normed, gated, streams, convWeight, state, { numTokens, channels, kernel, dilation }) {
  integer(numTokens, 'numTokens');
  integer(channels, 'channels');
  integer(kernel, 'kernel', 2);
  integer(dilation, 'dilation');
  const stateLength = (kernel - 1) * dilation;
  if (stateLength > 32) fail('the convolution state exceeds the kernel bound of 32 positions.');
  for (const [tensor, label] of /** @type {[import('../tensor.js').Tensor, string][]} */ ([[normed, 'normalized gated value'],
    [gated, 'gated value'], [streams, 'streams']])) {
    assertF32(tensor, [numTokens, channels], label);
  }
  assertF32(convWeight, [channels * kernel], 'convolution weight');
  if (!Number.isSafeInteger(state?.size) || state.size < stateLength * channels * 4) fail('convolution state buffer is too small.');
  if ([normed.buffer, gated.buffer, streams.buffer, convWeight.buffer].includes(state)) fail('state must not alias an input.');
  return withKernelOutput(target, null, numTokens * channels * 4, 'ple_output_streams', async (output) => {
    await unifiedKernelWrapper('ple_conv', target, 'f32', [normed, gated, streams, convWeight, state, output],
      { num_tokens: numTokens, channels, _pad0: 0, _pad1: 0 }, [Math.ceil(channels / WORKGROUP_SIZE), 1, 1],
      { WORKGROUP_SIZE, KERNEL: kernel, DILATION: dilation, STATE_LEN: stateLength });
    return createTensor(output, 'f32', [numTokens, channels], 'ple_output_streams');
  });
}

/** @type {typeof import('./ple.js').runPleRowDecode} */
export function runPleRowDecode(rows, options) { return executeDecode(null, rows, options); }
/** @type {typeof import('./ple.js').recordPleRowDecode} */
export function recordPleRowDecode(recorder, rows, options) { return executeDecode(recorder, rows, options); }
/** @type {typeof import('./ple.js').runPleGate} */
export function runPleGate(keyNormed, queryNormed, value, options) { return executeGate(null, keyNormed, queryNormed, value, options); }
/** @type {typeof import('./ple.js').recordPleGate} */
export function recordPleGate(recorder, keyNormed, queryNormed, value, options) {
  return executeGate(recorder, keyNormed, queryNormed, value, options);
}
/** @type {typeof import('./ple.js').runPleConv} */
export function runPleConv(normed, gated, streams, convWeight, state, options) {
  return executeConv(null, normed, gated, streams, convWeight, state, options);
}
/** @type {typeof import('./ple.js').recordPleConv} */
export function recordPleConv(recorder, normed, gated, streams, convWeight, state, options) {
  return executeConv(recorder, normed, gated, streams, convWeight, state, options);
}
