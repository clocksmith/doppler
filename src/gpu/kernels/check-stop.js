

import { getDevice, getDeviceEpoch } from '../device.js';
import { acquireBuffer, readBufferSlice } from '../../memory/buffer-pool.js';
import { recordDispatch } from './dispatch.js';
import { createUniformBufferFromData } from './uniform-utils.js';
import {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './pipeline-cache.js';
import { allowReadback } from '../perf-guards.js';
import { getShaderModule } from './shader-cache.js';


let checkStopPipeline = null;
let checkStopPipelineEpoch = -1;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

function getCheckStopBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'check_stop_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}


async function getCheckStopPipeline() {
  const epoch = getDeviceEpoch();
  if (checkStopPipeline && checkStopPipelineEpoch === epoch) return checkStopPipeline;
  const device = getDevice();
  const shaderModule = await getShaderModule(device, 'check_stop.wgsl', 'check_stop');
  const bindGroupLayout = getCheckStopBindGroupLayout(device);

  checkStopPipeline = device.createComputePipeline({
    layout: getOrCreatePipelineLayout('check_stop_pipeline_layout', [bindGroupLayout], device),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
      constants: { WORKGROUP_SIZE: 1 },
    },
  });
  checkStopPipelineEpoch = epoch;

  return checkStopPipeline;
}


export async function recordCheckStop(
  recorder,
  params
) {
  const device = getDevice();
  const pipeline = await getCheckStopPipeline();
  const tokenIndex = params.tokenIndex ?? 0;

  // Create uniform buffer
  const uniformData = new Uint32Array([
    params.eosTokenId,
    params.maxTokens,
    params.currentPos,
    tokenIndex,
  ]);
  const uniformBuffer = createUniformBufferFromData('check_stop_uniforms', uniformData, recorder);

  // Create output buffer
  const requiredBytes = (tokenIndex + 1) * U32_BYTES;
  const shouldStopBuffer = params.shouldStopBuffer ?? acquireBuffer(requiredBytes, undefined, 'check_stop_output');
  if (shouldStopBuffer.size < requiredBytes) {
    throw new Error('[CheckStop] shouldStopBuffer too small for tokenIndex.');
  }

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: getCheckStopBindGroupLayout(device),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.sampledTokenBuffer } },
      { binding: 2, resource: { buffer: shouldStopBuffer } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, 1, 'check_stop');

  return shouldStopBuffer;
}


export async function checkStop(params) {
  if (!allowReadback('check-stop')) {
    throw new Error('[CheckStop] GPU readback disabled');
  }

  const device = getDevice();
  const pipeline = await getCheckStopPipeline();

  const tokenIndex = params.tokenIndex ?? 0;
  const uniformData = new Uint32Array([
    params.eosTokenId,
    params.maxTokens,
    params.currentPos,
    tokenIndex,
  ]);
  const uniformBuffer = createUniformBufferFromData('check_stop_uniforms', uniformData, null, device);

  const requiredBytes = (tokenIndex + 1) * U32_BYTES;
  const shouldStopBuffer = params.shouldStopBuffer ?? device.createBuffer({
    size: requiredBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const ownsStopBuffer = !params.shouldStopBuffer;

  try {
    if (shouldStopBuffer.size < requiredBytes) {
      throw new Error('[CheckStop] shouldStopBuffer too small for tokenIndex.');
    }

    const bindGroup = device.createBindGroup({
      layout: getCheckStopBindGroupLayout(device),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: params.sampledTokenBuffer } },
        { binding: 2, resource: { buffer: shouldStopBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    device.queue.submit([encoder.finish()]);

    const result = new Uint32Array(
      await readBufferSlice(shouldStopBuffer, tokenIndex * U32_BYTES, U32_BYTES)
    )[0];
    return result === 1;
  } finally {
    uniformBuffer.destroy();
    if (ownsStopBuffer) {
      shouldStopBuffer.destroy();
    }
  }
}
