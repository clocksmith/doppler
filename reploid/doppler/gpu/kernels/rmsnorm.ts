/**
 * RMSNorm Kernels
 *
 * Provides RMS normalization with optional residual connection.
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

const DEBUG_KERNELS = typeof window !== 'undefined'
  ? Boolean((window as unknown as { DOPPLER_DEBUG_KERNELS?: boolean }).DOPPLER_DEBUG_KERNELS)
  : false;

/** RMSNorm kernel options */
export interface RMSNormOptions extends OutputBufferOptions {
  batchSize?: number;
  hiddenSize?: number | null;
  residual?: GPUBuffer | null;
}

/**
 * Select RMSNorm kernel variant
 */
export function selectRMSNormKernel(options: RMSNormOptions = {}): string {
  const { residual = null, hiddenSize = null } = options;
  if (residual) {
    return 'residual';
  } else if (hiddenSize !== null && hiddenSize <= 256) {
    return 'small';
  }
  return 'default';
}

/**
 * Run RMSNorm
 */
export async function runRMSNorm(
  input: GPUBuffer,
  weight: GPUBuffer,
  eps: number = 1e-5,
  options: RMSNormOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { batchSize = 1, hiddenSize, residual = null, outputBuffer = null } = options;

  // Select variant
  let variant = 'default';
  if (residual) {
    variant = 'residual';
    if (DEBUG_KERNELS) {
      console.log(`[RMSNorm] Using residual variant, residual.size=${residual.size}, inferredHiddenSize=${hiddenSize || (weight.size / 4)}, batchSize=${batchSize}`);
    }
  } else if (hiddenSize && hiddenSize <= 256) {
    variant = 'small';
  }

  const pipeline = await createPipeline('rmsnorm', variant);

  // Create output buffer if not provided
  const inferredHiddenSize = hiddenSize || (weight.size / 4);
  const outputSize = batchSize * inferredHiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

  // Create uniform buffer
  const hasResidualFlag = residual ? 1 : 0;
  const uniformBuffer = createUniformBufferWithView(
    'rmsnorm_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, hasResidualFlag, true); // hasResidual flag
    },
    null,
    device
  );
  if (DEBUG_KERNELS && hasResidualFlag) {
    console.log(`[RMSNorm] Uniform hasResidual=${hasResidualFlag}, hiddenSize=${inferredHiddenSize}, batchSize=${batchSize}`);
  }

  // Shader expects 5 bindings - create placeholder when no residual (uniform flags it as unused)
  const residualBuffer = residual || device.createBuffer({
    label: 'rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'rmsnorm_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residualBuffer } },
    ],
  });

  dispatch(device, pipeline, bindGroup, batchSize, 'rmsnorm');

  uniformBuffer.destroy();
  if (!residual) residualBuffer.destroy();

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Record RMSNorm (batched, no submit)
 */
export async function recordRMSNorm(
  recorder: CommandRecorder,
  input: GPUBuffer,
  weight: GPUBuffer,
  eps: number = 1e-5,
  options: RMSNormOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const {
    batchSize = 1,
    hiddenSize = null,
    residual = null,
    outputBuffer = null,
  } = options;

  // Infer hidden size from weight buffer
  const inferredHiddenSize = hiddenSize || (weight.size / 4);
  const inputSize = batchSize * inferredHiddenSize * 4;

  // Select kernel variant
  const variant = selectRMSNormKernel(options);
  const pipeline = await createPipeline('rmsnorm', variant);

  // Output buffer
  const output = outputBuffer || acquireBuffer(inputSize, undefined, 'rmsnorm_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'rmsnorm_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true); // hasResidual flag
    },
    recorder
  );

  // Shader expects 5 bindings - create placeholder when no residual (uniform flags it as unused)
  const residualBuffer = residual || device.createBuffer({
    label: 'rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'rmsnorm_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residualBuffer } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, batchSize, 'rmsnorm');

  // Track dummy buffer for cleanup if we created it
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }

  setBufferDtype(output, 'f32');
  return output;
}
