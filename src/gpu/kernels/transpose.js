import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';

function planTransposeDispatch(target, cols) {
  const device = target?.device;
  const maxPerDim = Number.isFinite(device?.limits?.maxComputeWorkgroupsPerDimension)
    ? device.limits.maxComputeWorkgroupsPerDimension
    : 65535;
  const dispatchStride = Math.min(cols, maxPerDim * WORKGROUP_SIZES.DEFAULT);
  return {
    dispatchStride,
    workgroups: [Math.ceil(dispatchStride / WORKGROUP_SIZES.DEFAULT), 1, 1],
  };
}

async function _transpose(target, input, rows, cols, options = {}) {
  const { outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = rows * cols * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'transpose_output');
  const dispatchPlan = planTransposeDispatch(target, cols);

  await unifiedKernelWrapper(
    'transpose',
    target,
    'default',
    [input, outputBuf],
    { rows, cols, _pad0: dispatchPlan.dispatchStride, _pad1: 0 },
    [dispatchPlan.workgroups[0], rows, 1]
  );

  return createTensor(outputBuf, input.dtype, [cols, rows], 'transpose_output');
}

export async function runTranspose(input, rows, cols, options = {}) {
  return _transpose(null, input, rows, cols, options);
}

export async function recordTranspose(recorder, input, rows, cols, options = {}) {
  return _transpose(recorder, input, rows, cols, options);
}
