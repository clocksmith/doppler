import { getDevice, getDeviceEpoch } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { recordDispatch } from './dispatch.js';
import { createUniformBufferFromData } from './uniform-utils.js';
import {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './pipeline-cache.js';
import { getShaderModule } from './shader-cache.js';

let pipeline = null;
let pipelineEpoch = -1;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

function getBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'check_hot_vocab_stop_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}

async function getPipeline() {
  const epoch = getDeviceEpoch();
  if (pipeline && pipelineEpoch === epoch) {
    return pipeline;
  }
  const device = getDevice();
  const shaderModule = await getShaderModule(device, 'check_hot_vocab_stop.wgsl', 'check_hot_vocab_stop');
  const bindGroupLayout = getBindGroupLayout(device);
  pipeline = device.createComputePipeline({
    layout: getOrCreatePipelineLayout('check_hot_vocab_stop_pipeline_layout', [bindGroupLayout], device),
    compute: {
      module: shaderModule,
      entryPoint: 'main',
      constants: { WORKGROUP_SIZE: 1 },
    },
  });
  pipelineEpoch = epoch;
  return pipeline;
}

export async function recordCheckHotVocabStop(recorder, params) {
  const device = getDevice();
  const hotStopPipeline = await getPipeline();
  const tokenIndex = params.tokenIndex ?? 0;
  const uniformData = new Uint32Array([
    params.eosTokenId,
    params.maxTokens,
    params.currentPos,
    tokenIndex,
    params.hotTokenSentinel,
    0,
    0,
    0,
  ]);
  const uniformBuffer = createUniformBufferFromData('check_hot_vocab_stop_uniforms', uniformData, recorder);
  const requiredBytes = (tokenIndex + 1) * U32_BYTES;
  const shouldStopBuffer = params.shouldStopBuffer ?? acquireBuffer(requiredBytes, undefined, 'check_hot_vocab_stop_output');
  if (shouldStopBuffer.size < requiredBytes) {
    throw new Error('[CheckHotVocabStop] shouldStopBuffer too small for tokenIndex.');
  }
  const bindGroup = device.createBindGroup({
    layout: getBindGroupLayout(device),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.sampledTokenBuffer } },
      { binding: 2, resource: { buffer: shouldStopBuffer } },
      { binding: 3, resource: { buffer: params.hotTokenIndexMapBuffer } },
      { binding: 4, resource: { buffer: params.nextInputTokenBuffer } },
    ],
  });
  recordDispatch(recorder, hotStopPipeline, bindGroup, 1, 'check_hot_vocab_stop');
  return shouldStopBuffer;
}
