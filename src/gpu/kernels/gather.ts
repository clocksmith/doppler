/**
 * Gather (Embedding Lookup) Kernels
 *
 * Provides token embedding lookups from embedding tables.
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView, getKernelConfig } from './utils.js';
import type { OutputBufferOptions } from './types.js';
import { trace } from '../../debug/index.js';
import { createTensor, type Tensor, type TensorDtype } from '../tensor.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';

// =============================================================================
// Variant Lookup Table
// =============================================================================

/**
 * Gather variant lookup table keyed by "${f16In}/${f16Out}/${vec4}".
 * Replaces if-else chain for variant selection.
 */
const GATHER_VARIANTS: Record<string, string> = {
  'false/false/false': 'default',
  'false/false/true': 'vec4',
  'true/false/false': 'f16',
  'true/false/true': 'f16_vec4',
  'false/true/false': 'f16_out',
  'false/true/true': 'vec4_f16_out',
  'true/true/false': 'f16_f16_out',
  'true/true/true': 'f16_vec4_f16_out',
};

/**
 * Select gather variant based on input/output dtype and vec4 preference.
 */
function selectGatherVariant(useF16Input: boolean, useF16Output: boolean, useVec4: boolean): string {
  const key = `${useF16Input}/${useF16Output}/${useVec4}`;
  const variant = GATHER_VARIANTS[key];
  if (!variant) {
    throw new Error(`Unknown gather variant combination: ${key}`);
  }
  return variant;
}

/**
 * Get output binding index from kernel config, falling back to 3 for F32 output.
 */
function getOutputBinding(variant: string, useF16Output: boolean): number {
  if (!useF16Output) {
    return 3; // F32 output always uses binding 3
  }
  const config = getKernelConfig('gather', variant);
  return config.variantMetadata?.outputBinding ?? 4;
}

/** Gather kernel options */
export interface GatherOptions extends OutputBufferOptions {
  useVec4?: boolean;
  embeddingDtype?: 'f16' | 'f32';  // Override auto-detection for input embeddings
  /**
   * Output dtype. When 'f16', converts F32 embeddings to F16 output.
   * Used for F16 activation mode to reduce memory bandwidth.
   * Default: 'f32'
   */
  outputDtype?: 'f16' | 'f32';
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
 * Returns Tensor with explicit dtype for type-safe pipeline.
 */
export async function runGather(
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options: GatherOptions = {}
): Promise<Tensor> {
  const device = getDevice();
  const { useVec4 = true, outputBuffer = null, embeddingDtype, outputDtype = 'f32', transpose = false } = options;

  // Detect embedding dtype (F16 embeddings enable optimized lm_head)
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || 'f32';
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;
  trace.embed(`Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, detectedDtype=${detectedDtype}, useF16Input=${useF16Input}, useF16Output=${useF16Output}`);

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = useF16Output ? 'f16' : 'f32';
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
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

  // Create bind group - output binding from kernel config
  const outputBinding = getOutputBinding(variant, useF16Output);
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } },
  ];
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // gather.wgsl: main uses @workgroup_size(256), gather_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads × 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'gather');

  uniformBuffer.destroy();

  const actualDtype: TensorDtype = useF16Output ? 'f16' : 'f32';
  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}

/**
 * Record gather (batched, no submit)
 * Returns Tensor with explicit dtype for type-safe pipeline.
 */
export async function recordGather(
  recorder: CommandRecorder,
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options: GatherOptions = {}
): Promise<Tensor> {
  const device = recorder.device;
  const { useVec4 = true, outputBuffer = null, embeddingDtype, outputDtype = 'f32', transpose = false } = options;

  // Detect embedding dtype (same logic as runGather)
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || 'f32';
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = useF16Output ? 'f16' : 'f32';
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
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

  // Create bind group - output binding from kernel config
  const outputBinding = getOutputBinding(variant, useF16Output);
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } },
  ];
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // gather.wgsl: main uses @workgroup_size(256), gather_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads × 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'gather');

  const actualDtype: TensorDtype = useF16Output ? 'f16' : 'f32';
  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}
