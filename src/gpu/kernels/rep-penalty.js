import { getDevice, getKernelCapabilities } from '../device.js';
import { createPipeline, getOrCreateBindGroupLayout } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { WORKGROUP_SIZES } from './constants.js';
import { selectRuleValue } from './rule-registry.js';


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

export function selectRepPenaltyVariant(useF16) {
  return selectRuleValue('repPenalty', 'variant', { useF16 });
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
    batchCount,
    batchOffset,
    logitsDtype,
  } = options;

  if (!Number.isFinite(batchCount) || batchCount < 0) {
    throw new Error('[RepPenalty] batchCount is required and must be non-negative.');
  }
  if (!Number.isFinite(batchOffset) || batchOffset < 0) {
    throw new Error('[RepPenalty] batchOffset is required and must be non-negative.');
  }
  if (logitsDtype !== 'f16' && logitsDtype !== 'f32') {
    throw new Error('[RepPenalty] logitsDtype must be "f16" or "f32".');
  }

  if (penalty === 1.0 || (historyCount === 0 && batchCount === 0)) {
    return;
  }

  const device = recorder.device;
  const useF16 = logitsDtype === 'f16';
  if (useF16 && !getKernelCapabilities()?.hasF16) {
    throw new Error('[RepPenalty] F16 logits requested but shader-f16 is unavailable.');
  }

  const variant = selectRepPenaltyVariant(useF16);
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
