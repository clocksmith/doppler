import { createTensor } from '../tensor.js';
import { isWeightBuffer } from '../weight-buffer.js';
import { getDevice } from '../device.js';
import { assertQ40Buffer, q40Dispatch } from './q4_0.js';
import { ggufBlockBytes } from '../../formats/gguf/block-storage.js';
import { withKernelOutput, unifiedKernelWrapper } from './kernel-execution.js';

/** @type {typeof import('./hyper-connection-up.js').runHyperConnectionUpMixBranch} */
export function runHyperConnectionUpMixBranch(input, weights, normed, options) {
  return execute(null, input, weights, normed, options);
}

/** @type {typeof import('./hyper-connection-up.js').recordHyperConnectionUpMixBranch} */
export function recordHyperConnectionUpMixBranch(recorder, input, weights, normed, options) {
  return execute(recorder, input, weights, normed, options);
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {unknown} weights
 * @param {import('../tensor.js').Tensor} normed
 * @param {import('./hyper-connection-up.js').HyperConnectionUpMixBranchOptions} options
 */
async function execute(recorder, input, weights, normed, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[HyperConnectionUp] An owning GPU device is required.');
  const { numTokens, streamCount, streamWidth, projectionRank, outputBuffer = null } = options;
  options.signal?.throwIfAborted();
  if (numTokens !== 1 || streamCount !== 4
    || ![streamWidth, projectionRank].every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || projectionRank % 32 !== 0 || streamWidth * streamCount * projectionRank > 0xffffffff
    || !(device.limits.maxComputeInvocationsPerWorkgroup >= 128)
    || !(device.limits.maxComputeWorkgroupSizeX >= 128) || !(device.limits.maxComputeWorkgroupSizeY >= 1)
    || !(device.limits.maxComputeWorkgroupSizeZ >= 1) || !(device.limits.maxComputeWorkgroupStorageSize >= 560)) {
    throw new Error('[HyperConnectionUp] Requires one token, four streams, whole Q4_0 blocks and supported gemv32 geometry.');
  }
  const width = streamCount * streamWidth;
  if (input.dtype !== 'f32' || input.precision != null || input.shape.length !== 2
    || input.shape[0] !== 1 || input.shape[1] !== projectionRank
    || normed.dtype !== 'f32' || normed.precision != null || normed.shape.length !== 2
    || normed.shape[0] !== 1 || normed.shape[1] !== width
    || !isWeightBuffer(weights) || weights.dtype !== 'q4_0' || weights.layout !== 'row'
    || weights.shape.length !== 2 || weights.shape[0] !== width || weights.shape[1] !== projectionRank) {
    throw new Error('[HyperConnectionUp] Requires explicit scalar F32 inputs and complete row-packed Q4_0 up weights; no dtype fallback.');
  }
  const usage = GPUBufferUsage.STORAGE;
  assertQ40Buffer(device, input.buffer, projectionRank * 4, usage, 'HC low-rank input');
  assertQ40Buffer(device, normed.buffer, width * 4, usage, 'HC normalized streams');
  assertQ40Buffer(device, weights.buffer, ggufBlockBytes(weights.shape, weights.dtype), usage, 'HC up weights');
  if (outputBuffer) {
    assertQ40Buffer(device, outputBuffer, streamWidth * 4, usage, 'HC branch output');
    if ([input.buffer, normed.buffer, weights.buffer].includes(outputBuffer)) {
      throw new Error('[HyperConnectionUp] Branch output must not alias an input.');
    }
  }
  return withKernelOutput(recorder, outputBuffer, streamWidth * 4, 'hyper_connection_up_branch', async output => {
    await unifiedKernelWrapper('hyper_connection_up_mix_branch', recorder, 'q4_0_gemv32_f32',
      [input.buffer, weights.buffer, normed.buffer, output],
      { stream_width: streamWidth, rank: projectionRank, blocks: projectionRank / 32, _pad0: 0 },
      q40Dispatch(device, streamWidth), null, null, 'hyper_connection_up_mix_branch', options.signal ?? null);
    return createTensor(output, 'f32', [1, streamWidth], 'hyper_connection_up_branch');
  });
}
