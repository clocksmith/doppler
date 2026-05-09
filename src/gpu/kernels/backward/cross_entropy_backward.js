import { dtypeBytes } from '../../tensor.js';
import { getDevice } from '../../device.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createBackwardKernel } from './utils.js';

function resolveMaxWorkgroupsPerDimension(device) {
  const limit = device?.limits?.maxComputeWorkgroupsPerDimension;
  return Number.isFinite(limit) && limit > 0 ? limit : 65535;
}

function planDispatch(numTokens, vocabSize, maxWorkgroupsPerDimension) {
  const total = numTokens * vocabSize;
  const dispatchStride = Math.min(
    total,
    maxWorkgroupsPerDimension * WORKGROUP_SIZES.DEFAULT
  );
  return {
    dispatchStride,
    workgroups: [
      Math.ceil(dispatchStride / WORKGROUP_SIZES.DEFAULT),
      Math.ceil(total / dispatchStride),
      1,
    ],
  };
}

const { run, record } = createBackwardKernel('cross_entropy_backward', {
  uniformSize: 16,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.numTokens, true);
    view.setUint32(4, opts.vocabSize, true);
    view.setUint32(8, opts._dispatchStride, true);
  },
  calcWorkgroups: (opts) => planDispatch(
    opts.numTokens,
    opts.vocabSize,
    opts._maxComputeWorkgroupsPerDimension
  ).workgroups,
  outputBytes: (opts) => opts.numTokens * opts.vocabSize * opts._bytesPerElement,
  outputShape: (opts) => [opts.numTokens, opts.vocabSize],
  dtype: (opts, inputs) => inputs[0].dtype,
  getDevice: true,
  validate: (opts) => {
    if (!opts.numTokens || !opts.vocabSize) throw new Error('cross entropy backward requires numTokens and vocabSize');
    const dispatchPlan = planDispatch(
      opts.numTokens,
      opts.vocabSize,
      opts._maxComputeWorkgroupsPerDimension
    );
    if (dispatchPlan.workgroups[1] > opts._maxComputeWorkgroupsPerDimension) {
      throw new Error(
        `cross entropy backward dispatch requires ${dispatchPlan.workgroups[1]} y workgroups ` +
        `but device limit is ${opts._maxComputeWorkgroupsPerDimension}`
      );
    }
  },
});

function withDispatchPlan(options, device) {
  const maxComputeWorkgroupsPerDimension = resolveMaxWorkgroupsPerDimension(device);
  const dispatchPlan = planDispatch(
    options.numTokens,
    options.vocabSize,
    maxComputeWorkgroupsPerDimension
  );
  return {
    ...options,
    _maxComputeWorkgroupsPerDimension: maxComputeWorkgroupsPerDimension,
    _dispatchStride: dispatchPlan.dispatchStride,
  };
}

export async function runCrossEntropyBackward(softmax, targets, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(softmax.dtype);
  return run(
    softmax,
    targets,
    gradOutput,
    {
      ...withDispatchPlan(options, getDevice()),
      _bytesPerElement: bytesPerElement,
    }
  );
}

export async function recordCrossEntropyBackward(recorder, softmax, targets, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(softmax.dtype);
  return record(
    recorder,
    softmax,
    targets,
    gradOutput,
    {
      ...withDispatchPlan(options, recorder.device),
      _bytesPerElement: bytesPerElement,
    }
  );
}
