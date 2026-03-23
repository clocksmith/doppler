


import { getDevice, getKernelCapabilities } from '../device.js';
import { createPipeline, getOrCreateBindGroupLayout } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { WORKGROUP_SIZES } from './constants.js';


function getRepPenaltyBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'rep_penalty_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
    device
  );
}

function resolveVariant(useF16) {
  return useF16 ? 'default_f16' : 'default';
}

function createRepPenaltyUniformBuffer(device, recorder, options) {
  return createUniformBufferWithView(
    'rep_penalty_uniforms',
    32,
    (view) => {
      view.setUint32(0, options.vocabSize, true);
      view.setUint32(4, options.historyCount, true);
      view.setFloat32(8, options.penalty, true);
      view.setUint32(12, options.batchCount, true);
      view.setUint32(16, options.batchOffset, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder,
    device
  );
}

export async function recordRepPenalty(
  recorder,
  logitsBuffer,
  historyBuffer,
  batchTokensBuffer,
  options
) {
  const {
    vocabSize,
    historyCount,
    penalty,
    batchCount = 0,
    batchOffset = 1,
    logitsDtype = 'f32',
  } = options;

  if (penalty === 1.0 || (historyCount === 0 && batchCount === 0)) {
    return;
  }

  const device = recorder.device;
  const useF16 = logitsDtype === 'f16';
  if (useF16 && !getKernelCapabilities()?.hasF16) {
    throw new Error('[RepPenalty] F16 logits requested but shader-f16 is unavailable.');
  }

  const variant = resolveVariant(useF16);
  const layout = getRepPenaltyBindGroupLayout(device);
  const pipeline = await createPipeline('rep_penalty', variant, layout);

  const uniformBuffer = createRepPenaltyUniformBuffer(device, recorder, {
    vocabSize,
    historyCount,
    penalty,
    batchCount,
    batchOffset,
  });

  const bindGroup = device.createBindGroup({
    label: 'rep_penalty_bind_group',
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logitsBuffer } },
      { binding: 2, resource: { buffer: historyBuffer } },
      { binding: 3, resource: { buffer: batchTokensBuffer } },
    ],
  });

  const totalTokens = historyCount + batchCount;
  const numWorkgroups = Math.ceil(totalTokens / WORKGROUP_SIZES.DEFAULT);

  const pass = recorder.beginComputePass('rep_penalty');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();
}
