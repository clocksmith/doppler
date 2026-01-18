

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { recordDispatch } from './dispatch.js';
import { createUniformBufferFromData, getOrCreateBindGroupLayout, getOrCreatePipelineLayout } from './utils.js';
import { allowReadback } from '../perf-guards.js';


let checkStopPipeline = null;

const SHADER = /* wgsl */ `
struct StopUniforms {
    eosTokenId: u32,
    maxTokens: u32,
    currentPos: u32,
    tokenIndex: u32,
}

struct ScalarU32 {
    value: u32,
}

@group(0) @binding(0) var<uniform> uniforms: StopUniforms;
@group(0) @binding(1) var<storage, read> sampledToken: array<u32>;
@group(0) @binding(2) var<storage, read_write> shouldStop: ScalarU32;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let token = sampledToken[uniforms.tokenIndex];
    let isEOS = (token == uniforms.eosTokenId);
    let reachedMax = (uniforms.currentPos >= uniforms.maxTokens);

    if (isEOS || reachedMax) {
        shouldStop.value = 1u;
    } else {
        shouldStop.value = 0u;
    }
}
`;


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


function getCheckStopPipeline() {
  if (checkStopPipeline) return checkStopPipeline;

  const device = getDevice();
  const shaderModule = device.createShaderModule({ code: SHADER });
  const bindGroupLayout = getCheckStopBindGroupLayout(device);

  checkStopPipeline = device.createComputePipeline({
    layout: getOrCreatePipelineLayout('check_stop_pipeline_layout', [bindGroupLayout], device),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  return checkStopPipeline;
}


export function recordCheckStop(
  recorder,
  params
) {
  const device = getDevice();
  const pipeline = getCheckStopPipeline();

  // Create uniform buffer
  const uniformData = new Uint32Array([
    params.eosTokenId,
    params.maxTokens,
    params.currentPos,
    params.tokenIndex ?? 0,
  ]);
  const uniformBuffer = createUniformBufferFromData('check_stop_uniforms', uniformData, recorder);

  // Create output buffer
  const shouldStopBuffer = acquireBuffer(4, undefined, 'check_stop_output');

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
  const pipeline = getCheckStopPipeline();

  const uniformData = new Uint32Array([
    params.eosTokenId,
    params.maxTokens,
    params.currentPos,
    params.tokenIndex ?? 0,
  ]);
  const uniformBuffer = createUniformBufferFromData('check_stop_uniforms', uniformData, null, device);

  const shouldStopBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

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

  // Readback result
  const stagingBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(shouldStopBuffer, 0, stagingBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  uniformBuffer.destroy();
  shouldStopBuffer.destroy();
  stagingBuffer.destroy();

  return result === 1;
}
