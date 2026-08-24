import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { getBuffer, isGpuBufferInstance, isWeightBuffer } from '../weight-buffer.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

const WORKGROUP_SIZE = 256;

function resolveBiasBuffer(device, bias, sourceColumns) {
  if (bias == null) {
    const buffer = acquireBuffer(4, undefined, 'logit_finalize_zero_bias');
    device.queue.writeBuffer(buffer, 0, new Float32Array(1));
    return { buffer, owned: true, hasBias: false };
  }
  if (bias instanceof Float32Array) {
    if (bias.length < sourceColumns) {
      throw new Error(
        `[LogitFinalize] bias requires at least ${sourceColumns} values, got ${bias.length}.`
      );
    }
    const buffer = acquireBuffer(bias.byteLength, undefined, 'logit_finalize_bias');
    device.queue.writeBuffer(buffer, 0, bias);
    return { buffer, owned: true, hasBias: true };
  }
  if (isWeightBuffer(bias)) {
    return { buffer: getBuffer(bias), owned: false, hasBias: true };
  }
  if (isGpuBufferInstance(bias)) {
    return { buffer: bias, owned: false, hasBias: true };
  }
  throw new Error('[LogitFinalize] bias must be null, Float32Array, GPUBuffer, or WeightBuffer.');
}

async function finalizeLogitsTensor(target, input, options) {
  const device = getDevice();
  if (!device) {
    throw new Error('[LogitFinalize] GPU device is required.');
  }
  const rowCount = Number(options?.rowCount);
  const sourceColumns = Number(options?.sourceColumns);
  const targetColumns = Number(options?.targetColumns);
  const outputScale = Number(options?.outputScale);
  const softcap = Number(options?.softcap);
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error('[LogitFinalize] rowCount must be a positive integer.');
  }
  if (!Number.isInteger(sourceColumns) || sourceColumns < 1) {
    throw new Error('[LogitFinalize] sourceColumns must be a positive integer.');
  }
  if (!Number.isInteger(targetColumns) || targetColumns < sourceColumns) {
    throw new Error('[LogitFinalize] targetColumns must be an integer not smaller than sourceColumns.');
  }
  if (!Number.isFinite(outputScale)) {
    throw new Error('[LogitFinalize] outputScale must be finite.');
  }
  if (!Number.isFinite(softcap) || softcap < 0) {
    throw new Error('[LogitFinalize] softcap must be finite and non-negative.');
  }
  if (input?.dtype !== 'f16' && input?.dtype !== 'f32') {
    throw new Error(`[LogitFinalize] unsupported input dtype "${input?.dtype ?? 'unknown'}".`);
  }
  const output = acquireBuffer(
    rowCount * targetColumns * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    'logit_finalize_output'
  );
  let bias = null;
  try {
    bias = resolveBiasBuffer(device, options.bias ?? null, sourceColumns);
    await unifiedKernelWrapper(
      'logit_finalize',
      target,
      input.dtype,
      [input, bias.buffer, output],
      {
        row_count: rowCount,
        source_columns: sourceColumns,
        target_columns: targetColumns,
        has_bias: bias.hasBias ? 1 : 0,
        output_scale: outputScale,
        softcap,
      },
      Math.ceil((rowCount * targetColumns) / WORKGROUP_SIZE)
    );
    return createTensor(output, 'f32', [rowCount, targetColumns], 'logit_finalize_output');
  } catch (error) {
    releaseBuffer(output);
    throw error;
  } finally {
    if (bias?.owned) {
      if (target) target.trackTemporaryBuffer(bias.buffer);
      else releaseBuffer(bias.buffer);
    }
  }
}

export async function runFinalizeLogitsTensor(input, options) {
  return finalizeLogitsTensor(null, input, options);
}

export async function recordFinalizeLogitsTensor(recorder, input, options) {
  return finalizeLogitsTensor(recorder, input, options);
}
