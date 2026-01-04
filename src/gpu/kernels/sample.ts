/**
 * GPU-Side Sampling Kernel
 *
 * Performs sampling entirely on GPU, reducing readback from ~1MB to 4 bytes.
 * Supports:
 * - Temperature scaling
 * - Top-k selection
 * - Softmax
 * - Multinomial sampling
 * - Greedy argmax (for temperature=0)
 */

import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../buffer-pool.js';
import { WORKGROUP_SIZES } from './constants.js';
import { createPipeline, createUniformBufferWithView, getOrCreateBindGroupLayout } from './utils.js';
import { allowReadback } from '../perf-guards.js';
import type { CommandRecorder } from '../command-recorder.js';
import { DEFAULT_SAMPLING_DEFAULTS } from '../../config/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';

export interface SampleOptions {
  temperature?: number;
  topK?: number;
  randomSeed?: number;
  padTokenId?: number;
  logitSoftcap?: number;  // Gemma 2: 30.0, 0 = disabled
}

export interface SampleResult {
  tokenId: number;
  gpuBuffer: GPUBuffer;  // Buffer containing the token ID
}

/**
 * Get or create explicit bind group layout for sample kernels.
 * Required because different entry points use different binding subsets,
 * so layout: 'auto' fails to include all bindings.
 */
function getSampleBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  return getOrCreateBindGroupLayout(
    'sample_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}

/**
 * Create sample pipeline with explicit bind group layout.
 */
async function createSamplePipeline(device: GPUDevice, entryPoint: string): Promise<GPUComputePipeline> {
  return createPipeline('sample', entryPoint, getSampleBindGroupLayout(device));
}

/**
 * Run GPU-side argmax (greedy decoding)
 * Returns the token ID with highest logit
 */
export async function runArgmax(
  logits: GPUBuffer,
  vocabSize: number,
  options: SampleOptions = {}
): Promise<number> {
  if (!allowReadback('sample.runArgmax')) {
    throw new Error('[Sample] GPU readback disabled for argmax');
  }

  const device = getDevice();
  if (!device) throw new Error('GPU device not initialized');

  // Pipelines with explicit layout
  const argmaxPipeline = await createSamplePipeline(device, 'argmax');
  const reducePipeline = await createSamplePipeline(device, 'argmax_reduce');

  // Workgroups for first pass
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));

  // Intermediate buffers
  const tempLogits = acquireBuffer(workgroupSize * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(workgroupSize * 4, undefined, 'argmax_temp_indices');
  const outputBuffer = acquireBuffer(4, undefined, 'argmax_output');

  // Uniforms
  const padTokenId = options.padTokenId ?? 0xFFFFFFFF;
  const logitSoftcap = options.logitSoftcap ?? 0;
  const uniformBuffer = createUniformBufferWithView(
    'argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);     // vocabSize
      view.setUint32(4, 1, true);             // topK (unused for argmax)
      view.setFloat32(8, 1.0, true);          // temperature (unused)
      view.setFloat32(12, 0.0, true);         // randomValue (unused)
      view.setUint32(16, padTokenId, true);   // padTokenId
      view.setFloat32(20, logitSoftcap, true); // logitSoftcap (Gemma 2: 30.0)
    },
    null,
    device
  );

  // Bind groups with explicit layout (auto-layout fails for multi-entry-point shaders)
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const argmaxBindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  const reduceBindGroup = device.createBindGroup({
    label: 'argmax_reduce_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },  // Shader may not use, but layout requires
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  // Execute
  const encoder = device.createCommandEncoder({ label: 'argmax_encoder' });

  // Pass 1: Find max per workgroup
  const pass1 = encoder.beginComputePass({ label: 'argmax_pass1' });
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, argmaxBindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Pass 2: Reduce workgroup results
  const pass2 = encoder.beginComputePass({ label: 'argmax_pass2' });
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  device.queue.submit([encoder.finish()]);

  // Read result
  const stagingBuffer = device.createBuffer({
    label: 'argmax_staging',
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({ label: 'argmax_copy' });
  copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
  device.queue.submit([copyEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  // Cleanup
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(tempLogits);
  releaseBuffer(tempIndices);
  releaseBuffer(outputBuffer);

  return tokenId;
}

/**
 * Run GPU-side top-k sampling
 * Applies temperature, selects top-k, applies softmax, samples
 */
export async function runGPUSample(
  logits: GPUBuffer,
  vocabSize: number,
  options: SampleOptions = {}
): Promise<number> {
  if (!allowReadback('sample.runGPUSample')) {
    throw new Error('[Sample] GPU readback disabled for sampling');
  }

  const {
    temperature = DEFAULT_SAMPLING_DEFAULTS.temperature,
    topK = DEFAULT_SAMPLING_DEFAULTS.topK,
    randomSeed,
    padTokenId,
    logitSoftcap = 0,
  } = options;

  // For temperature=0 or very low, use greedy argmax
  const { greedyThreshold } = getRuntimeConfig().inference.sampling;
  if (temperature < greedyThreshold) {
    return runArgmax(logits, vocabSize, { padTokenId, logitSoftcap });
  }

  const device = getDevice();
  if (!device) throw new Error('GPU device not initialized');

  // Generate random value for sampling
  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  // Get pipelines with explicit layout
  const phase1Pipeline = await createSamplePipeline(device, 'find_topk_phase1');
  const phase2Pipeline = await createSamplePipeline(device, 'find_topk_phase2');
  const phase3Pipeline = await createSamplePipeline(device, 'softmax_and_sample');

  // Workgroups for phase 1
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));

  // Buffers
  const topkLogits = acquireBuffer(workgroupSize * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(workgroupSize * 4, undefined, 'topk_indices');
  const outputBuffer = acquireBuffer(4, undefined, 'sample_output');

  // Uniforms
  const uniformBuffer = createUniformBufferWithView(
    'sample_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId ?? 0xFFFFFFFF, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
    },
    null,
    device
  );

  // Bind group with explicit layout (auto-layout fails for multi-entry-point shaders)
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'sample_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: topkIndices } },
      { binding: 4, resource: { buffer: topkLogits } },
    ],
  });

  // Execute all phases
  const encoder = device.createCommandEncoder({ label: 'sample_encoder' });

  // Phase 1: Find per-workgroup top values
  const pass1 = encoder.beginComputePass({ label: 'sample_phase1' });
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Phase 2: Merge and select top-k
  const pass2 = encoder.beginComputePass({ label: 'sample_phase2' });
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Phase 3: Softmax and sample
  const pass3 = encoder.beginComputePass({ label: 'sample_phase3' });
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  device.queue.submit([encoder.finish()]);

  // Read result (just 4 bytes!)
  const stagingBuffer = device.createBuffer({
    label: 'sample_staging',
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({ label: 'sample_copy' });
  copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
  device.queue.submit([copyEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  // Cleanup
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(topkLogits);
  releaseBuffer(topkIndices);
  releaseBuffer(outputBuffer);

  return tokenId;
}

/**
 * Record GPU argmax (batched, no submit)
 * Returns buffer containing token ID
 */
export async function recordArgmax(
  recorder: CommandRecorder,
  logits: GPUBuffer,
  vocabSize: number,
  options: SampleOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;

  // Pipelines with explicit layout
  const argmaxPipeline = await createSamplePipeline(device, 'argmax');
  const reducePipeline = await createSamplePipeline(device, 'argmax_reduce');

  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  // Buffers
  const tempLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_indices');
  const outputBuffer = acquireBuffer(4, undefined, 'argmax_output');

  // Uniforms
  const padTokenId = options.padTokenId ?? 0xFFFFFFFF;
  const logitSoftcap = options.logitSoftcap ?? 0;
  const uniformBuffer = createUniformBufferWithView(
    'argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1.0, true);
      view.setFloat32(12, 0.0, true);
      view.setUint32(16, padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
    },
    recorder
  );

  // Bind groups with explicit layout
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  // Pass 1
  const pass1 = recorder.beginComputePass('argmax_phase1');
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Pass 2 (reuse same bind group since layout is the same)
  const reduceBindGroup = device.createBindGroup({
    label: 'argmax_reduce_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  const pass2 = recorder.beginComputePass('argmax_phase2');
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Schedule cleanup of temp buffers after submit
  // (These will be released by caller after reading output)

  return outputBuffer;
}

/**
 * Record GPU top-k sampling (batched, no submit)
 * Returns buffer containing token ID
 */
export async function recordGPUSample(
  recorder: CommandRecorder,
  logits: GPUBuffer,
  vocabSize: number,
  options: SampleOptions = {}
): Promise<GPUBuffer> {
  const {
    temperature = DEFAULT_SAMPLING_DEFAULTS.temperature,
    topK = DEFAULT_SAMPLING_DEFAULTS.topK,
    randomSeed,
    padTokenId,
    logitSoftcap = 0,
  } = options;

  // For temperature=0 or very low, use greedy argmax
  const { greedyThreshold } = getRuntimeConfig().inference.sampling;
  if (temperature < greedyThreshold) {
    return recordArgmax(recorder, logits, vocabSize, { padTokenId, logitSoftcap });
  }

  const device = recorder.device;

  // Generate random value for sampling
  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  // Get pipelines with explicit layout
  const phase1Pipeline = await createSamplePipeline(device, 'find_topk_phase1');
  const phase2Pipeline = await createSamplePipeline(device, 'find_topk_phase2');
  const phase3Pipeline = await createSamplePipeline(device, 'softmax_and_sample');

  // Workgroups for phase 1
  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  // Buffers
  const topkLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_indices');
  const outputBuffer = acquireBuffer(4, undefined, 'sample_output');

  // Uniforms
  const uniformBuffer = createUniformBufferWithView(
    'sample_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId ?? 0xFFFFFFFF, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
    },
    recorder
  );

  // Bind group with explicit layout
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'sample_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: topkIndices } },
      { binding: 4, resource: { buffer: topkLogits } },
    ],
  });

  // Phase 1: Find per-workgroup top values
  const pass1 = recorder.beginComputePass('sample_phase1');
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Phase 2: Merge and select top-k
  const pass2 = recorder.beginComputePass('sample_phase2');
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Phase 3: Softmax and sample
  const pass3 = recorder.beginComputePass('sample_phase3');
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  // Track temp buffers for cleanup
  recorder.trackTemporaryBuffer(topkLogits);
  recorder.trackTemporaryBuffer(topkIndices);

  return outputBuffer;
}

/**
 * Simple seeded random number generator
 */
function seededRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

/**
 * Check if GPU sampling is available
 */
export function isGPUSamplingAvailable(): boolean {
  return getDevice() !== null;
}
