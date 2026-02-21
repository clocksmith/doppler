

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { trace, isTraceEnabled } from '../../debug/index.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { isFusedQ4KDisabled } from './matmul.js';
import { getKernelThresholds, QK_K } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';

class FusedFFNKernel extends KernelBase {

  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('fused_ffn', variant, null, constants);
  }


  dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }


  record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }
}


function selectFFNVariant(batchSize, weightDtype, intermediateSize, hiddenSize) {
  const { multiOutputThreshold } = getKernelThresholds().ffn;
  const capabilities = getKernelCapabilities();
  const isQ4K = weightDtype === 'q4k';
  const fusedAllowed = !isFusedQ4KDisabled();
  const hiddenAligned = hiddenSize % QK_K === 0;
  const useMultiOutput = intermediateSize <= multiOutputThreshold;
  const hasF16 = capabilities.hasF16;

  return selectRuleValue(
    'fusedFfn',
    'variant',
    { isQ4K, fusedAllowed, hiddenAligned, batchSize, weightDtype, useMultiOutput, hasF16 }
  );
}


function createFFNUniformBuffer(device, recorder, params) {
  const swigluLimit = resolveSwigluLimit(params.swigluLimit, 'FusedFFN uniforms');
  return createUniformBufferWithView(
    'fused_ffn_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === 'silu' ? 0 : 1, true);
      // Q4K needs num_blocks_per_row at offset 20
      if (params.isQ4K) {
        view.setUint32(20, Math.floor(params.hiddenSize / 256), true);
      }
      view.setFloat32(24, swigluLimit, true);
    },
    recorder,
    device
  );
}

function resolveSwigluLimit(value, context) {
  if (value === undefined) {
    throw new Error(`${context} requires an explicit swigluLimit (null or number).`);
  }
  if (value == null) return 0;
  return value;
}


export async function runFusedFFN(
  input,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const device = getDevice();
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
    swigluLimit,
  } = options;
  resolveSwigluLimit(swigluLimit, 'FusedFFN');

  const gateDtype = getWeightDtype(W_gate);
  const upDtype = getWeightDtype(W_up);
  if (!gateDtype || !upDtype) {
    throw new Error('Fused FFN requires explicit gate/up weight dtypes');
  }
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== 'f16' && gateDtype !== 'f32' && gateDtype !== 'q4k') {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }

  const isQ4K = gateDtype === 'q4k';
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize, hiddenSize);
  const isF16Native = variant === 'f16_native' || variant === 'f16_native_batched';

  if (isF16Native) {
    if (input.dtype !== 'f16') {
      throw new Error('f16_native FFN variant requires f16 activations');
    }
  } else if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  trace.kernels(`FusedFFN: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const constants = (hiddenSize % 256 !== 0 && hiddenSize % 128 === 0)
    ? { SHARED_INPUT_SIZE: 128 }
    : null;
  const pipeline = await kernel.getPipeline(variant, constants);

  // Create output buffer: f16_native outputs f16 (2 bytes), others output f32 (4 bytes)
  const outputBytesPerElement = isF16Native ? 2 : 4;
  const outputDtype = isF16Native ? 'f16' : 'f32';
  const outputSize = batchSize * intermediateSize * outputBytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_ffn_output');

  // Create uniform buffer
  const uniformBuffer = createFFNUniformBuffer(device, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K,
    swigluLimit: activation === 'silu' ? swigluLimit : null,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'fused_ffn_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  // Calculate workgroups

  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'q4k' || variant === 'q4k_batched') {
    // Q4K uses multi-column: 32 columns per workgroup
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === 'q4k_batched' ? batchSize : 1;
  } else if (variant === 'batched' || variant === 'f16_native_batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);

  uniformBuffer.destroy();

  return createTensor(output, outputDtype, [batchSize, intermediateSize], 'fused_ffn_output');
}


export async function recordFusedFFN(
  recorder,
  input,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const device = recorder.device;
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
    swigluLimit,
  } = options;
  resolveSwigluLimit(swigluLimit, 'FusedFFN');

  const gateDtype = getWeightDtype(W_gate);
  const upDtype = getWeightDtype(W_up);
  if (!gateDtype || !upDtype) {
    throw new Error('Fused FFN requires explicit gate/up weight dtypes');
  }
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== 'f16' && gateDtype !== 'f32' && gateDtype !== 'q4k') {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }

  const isQ4K = gateDtype === 'q4k';
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize, hiddenSize);
  const isF16Native = variant === 'f16_native' || variant === 'f16_native_batched';

  if (isF16Native) {
    if (input.dtype !== 'f16') {
      throw new Error('f16_native FFN variant requires f16 activations');
    }
  } else if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  trace.kernels(`FusedFFN record: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const constants = (hiddenSize % 256 !== 0 && hiddenSize % 128 === 0)
    ? { SHARED_INPUT_SIZE: 128 }
    : null;
  const pipeline = await kernel.getPipeline(variant, constants);

  const outputBytesPerElement = isF16Native ? 2 : 4;
  const outputDtype = isF16Native ? 'f16' : 'f32';
  const outputSize = batchSize * intermediateSize * outputBytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_ffn_output');

  const uniformBuffer = createFFNUniformBuffer(device, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K,
    swigluLimit: activation === 'silu' ? swigluLimit : null,
  });

  const bindGroup = device.createBindGroup({
    label: 'fused_ffn_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } },
    ],
  });


  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'q4k' || variant === 'q4k_batched') {
    // Q4K uses multi-column: 32 columns per workgroup
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === 'q4k_batched' ? batchSize : 1;
  } else if (variant === 'batched' || variant === 'f16_native_batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);

  return createTensor(output, outputDtype, [batchSize, intermediateSize], 'fused_ffn_output');
}


export function calculateFusedFFNSavings(
  batchSize,
  hiddenSize,
  intermediateSize
) {
  // Separate kernel approach:
  // - Read input 2x (once for gate, once for up)
  // - Write gate output, up output, final output
  const inputBytes = batchSize * hiddenSize * 4;
  const intermediateBytes = batchSize * intermediateSize * 4;
  const separateBytes = 2 * inputBytes + 3 * intermediateBytes;

  // Fused approach:
  // - Read input 1x
  // - Write final output 1x
  const fusedBytes = inputBytes + intermediateBytes;

  const savingsBytes = separateBytes - fusedBytes;
  const savingsPct = (savingsBytes / separateBytes) * 100;

  return {
    separateBytes,
    fusedBytes,
    savingsBytes,
    savingsPct,
  };
}
