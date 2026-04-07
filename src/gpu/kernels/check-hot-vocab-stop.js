import { getDevice, getDeviceEpoch } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { recordDispatch } from './dispatch.js';
import { createUniformBufferFromData, getOrCreateBindGroupLayout, getOrCreatePipelineLayout } from './utils.js';

let pipeline = null;
let pipelineEpoch = -1;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

const SHADER = /* wgsl */ `
override WORKGROUP_SIZE: u32 = 1u;

struct HotStopUniforms {
    eosTokenId: u32,
    maxTokens: u32,
    currentPos: u32,
    tokenIndex: u32,
    hotTokenSentinel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: HotStopUniforms;
@group(0) @binding(1) var<storage, read> sampledToken: array<u32>;
@group(0) @binding(2) var<storage, read_write> shouldStop: array<u32>;
@group(0) @binding(3) var<storage, read> hotTokenIndexMap: array<u32>;
@group(0) @binding(4) var<storage, read_write> nextInputToken: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main() {
    let token = sampledToken[uniforms.tokenIndex];
    let hotIndex = hotTokenIndexMap[token];
    let hotMiss = hotIndex == uniforms.hotTokenSentinel;
    let isEOS = token == uniforms.eosTokenId;
    let reachedMax = uniforms.currentPos >= uniforms.maxTokens;
    shouldStop[uniforms.tokenIndex] = select(0u, 1u, hotMiss || isEOS || reachedMax);
    nextInputToken[uniforms.tokenIndex] = hotIndex;
}
`;

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

function getPipeline() {
  const epoch = getDeviceEpoch();
  if (pipeline && pipelineEpoch === epoch) {
    return pipeline;
  }
  const device = getDevice();
  const shaderModule = device.createShaderModule({ code: SHADER });
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

export function recordCheckHotVocabStop(recorder, params) {
  const device = getDevice();
  const hotStopPipeline = getPipeline();
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
