/**
 * Gather (Embedding Lookup) Kernels
 *
 * Provides token embedding lookups from embedding tables.
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { getBufferDtype, setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** Gather kernel options */
export interface GatherOptions extends OutputBufferOptions {
  useVec4?: boolean;
  embeddingDtype?: 'f16' | 'f32';  // Override auto-detection
  /**
   * True if embeddings are stored as [hidden_size, vocab_size] (GGUF layout).
   * False if embeddings are stored as [vocab_size, hidden_size] (PyTorch layout).
   * Default: false (RDRR format uses PyTorch layout from SafeTensors).
   */
  transpose?: boolean;
}

/**
 * Run gather/embedding lookup
 * Automatically detects F16 embeddings and uses optimized kernel
 */
export async function runGather(
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options: GatherOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { useVec4 = true, outputBuffer = null, embeddingDtype, transpose = false } = options;

  // Detect embedding dtype (F16 embeddings enable optimized lm_head)
  const caps = getKernelCapabilities();
  const bufferDtype = getBufferDtype(embeddings);
  const detectedDtype = embeddingDtype || bufferDtype || 'f32';
  const useF16 = detectedDtype === 'f16' && caps.hasF16;
  console.log(`[Gather] numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, bufferDtype=${bufferDtype}, detectedDtype=${detectedDtype}, useF16=${useF16}`);

  // Select kernel variant based on dtype and vec4 preference
  let variant: string;
  if (useF16) {
    variant = useVec4 ? 'f16_vec4' : 'f16';
  } else {
    variant = useVec4 ? 'vec4' : 'default';
  }
  const pipeline = await createPipeline('gather', variant);

  const outputSize = numTokens * hiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gather_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: embeddings } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  // gather.wgsl: main uses @workgroup_size(256), gather_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads × 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / 256)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'gather');

  uniformBuffer.destroy();

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Record gather (batched, no submit)
 */
export async function recordGather(
  recorder: CommandRecorder,
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options: GatherOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { useVec4 = true, outputBuffer = null, embeddingDtype, transpose = false } = options;

  // Detect embedding dtype (same logic as runGather)
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || getBufferDtype(embeddings) || 'f32';
  const useF16 = detectedDtype === 'f16' && caps.hasF16;

  // Select kernel variant based on dtype and vec4 preference
  let variant: string;
  if (useF16) {
    variant = useVec4 ? 'f16_vec4' : 'f16';
  } else {
    variant = useVec4 ? 'vec4' : 'default';
  }
  const pipeline = await createPipeline('gather', variant);

  const outputSize = numTokens * hiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gather_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: embeddings } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  // gather.wgsl: main uses @workgroup_size(256), gather_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads × 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / 256)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'gather');

  setBufferDtype(output, 'f32');
  return output;
}
