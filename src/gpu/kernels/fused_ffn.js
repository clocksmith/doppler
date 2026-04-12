

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { FFN_DISPATCH } from './constants.js';
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

const SHARED_INPUT_SIZE_VARIANTS = new Set([
  'default',
  'batched',
  'f16',
  'multi',
  'f16_native',
  'f16_native_batched',
]);

const F16_INPUT_VARIANTS = new Set([
  'f16_native',
  'f16_native_batched',
  'q4k_f16a',
  'q4k_batched_f16a',
]);

const F16_OUTPUT_VARIANTS = new Set([
  'f16_native',
  'f16_native_batched',
]);


function selectFFNVariant(batchSize, weightDtype, intermediateSize, hiddenSize, inputDtype) {
  const { multiOutputThreshold } = getKernelThresholds().ffn;
  const capabilities = getKernelCapabilities();
  const isQ4K = weightDtype === 'q4k';
  const fusedAllowed = !isFusedQ4KDisabled();
  const hiddenSubblockAligned = hiddenSize % 32 === 0;
  const useMultiOutput = intermediateSize <= multiOutputThreshold;
  const hasF16 = capabilities.hasF16;
  const useF16Input = inputDtype === 'f16';

  return selectRuleValue(
    'fusedFfn',
    'variant',
    {
      isQ4K,
      fusedAllowed,
      hiddenSubblockAligned,
      batchSize,
      weightDtype,
      useMultiOutput,
      hasF16,
      useF16Input,
    }
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
        view.setUint32(20, Math.ceil(params.hiddenSize / 256), true);
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

function calculateFFNDispatch(variant, batchSize, intermediateSize) {
  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    workgroupsX = Math.ceil(intermediateSize / FFN_DISPATCH.MULTI_OUTPUTS_PER_WG);
  } else if (
    variant === 'q4k'
    || variant === 'q4k_batched'
    || variant === 'q4k_f16a'
    || variant === 'q4k_batched_f16a'
  ) {
    workgroupsX = Math.ceil(intermediateSize / FFN_DISPATCH.Q4K_COLS_PER_WG);
    workgroupsY = (variant === 'q4k_batched' || variant === 'q4k_batched_f16a') ? batchSize : 1;
  } else if (variant === 'batched' || variant === 'f16_native_batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  return { workgroupsX, workgroupsY };
}

function resolveFusedFFNPipelineConstants(variant, hiddenSize) {
  if (!SHARED_INPUT_SIZE_VARIANTS.has(variant)) {
    return null;
  }
  return (hiddenSize % FFN_DISPATCH.SHARED_INPUT_SIZE_DEFAULT !== 0 &&
      hiddenSize % FFN_DISPATCH.SHARED_INPUT_SIZE_SMALL === 0)
    ? { SHARED_INPUT_SIZE: FFN_DISPATCH.SHARED_INPUT_SIZE_SMALL }
    : null;
}


function releaseRunResources(uniformBuffer, ownedBuffers) {
  if (uniformBuffer) {
    uniformBuffer.destroy();
  }
  for (const buffer of ownedBuffers) {
    if (buffer) {
      releaseBuffer(buffer);
    }
  }
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
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize, hiddenSize, input.dtype);
  const requiresF16Input = F16_INPUT_VARIANTS.has(variant);
  const usesF16Output = F16_OUTPUT_VARIANTS.has(variant);

  if (requiresF16Input) {
    if (input.dtype !== 'f16') {
      throw new Error(`Fused FFN variant ${variant} requires f16 activations`);
    }
  } else if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  trace.kernels(`FusedFFN: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const constants = resolveFusedFFNPipelineConstants(variant, hiddenSize);
  const pipeline = await kernel.getPipeline(variant, constants);

  // Native f16 weight kernels narrow back to f16 output. Q4K f16-activation
  // variants keep the existing f32 output contract so downstream precision is
  // still controlled explicitly by the caller/kernel path.
  const outputBytesPerElement = usesF16Output ? 2 : 4;
  const outputDtype = usesF16Output ? 'f16' : 'f32';
  const outputSize = batchSize * intermediateSize * outputBytesPerElement;
  const ownedOutput = outputBuffer ? null : acquireBuffer(outputSize, undefined, 'fused_ffn_output');
  const output = outputBuffer || ownedOutput;

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

  try {
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

    const { workgroupsX, workgroupsY } = calculateFFNDispatch(variant, batchSize, intermediateSize);
    kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);
  } catch (error) {
    releaseRunResources(uniformBuffer, [ownedOutput]);
    throw error;
  }

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
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize, hiddenSize, input.dtype);
  const requiresF16Input = F16_INPUT_VARIANTS.has(variant);
  const usesF16Output = F16_OUTPUT_VARIANTS.has(variant);

  if (requiresF16Input) {
    if (input.dtype !== 'f16') {
      throw new Error(`Fused FFN variant ${variant} requires f16 activations`);
    }
  } else if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  trace.kernels(`FusedFFN record: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const constants = resolveFusedFFNPipelineConstants(variant, hiddenSize);
  const pipeline = await kernel.getPipeline(variant, constants);

  const outputBytesPerElement = usesF16Output ? 2 : 4;
  const outputDtype = usesF16Output ? 'f16' : 'f32';
  const outputSize = batchSize * intermediateSize * outputBytesPerElement;
  const ownedOutput = outputBuffer ? null : acquireBuffer(outputSize, undefined, 'fused_ffn_output');
  const output = outputBuffer || ownedOutput;

  const uniformBuffer = createFFNUniformBuffer(device, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K,
    swigluLimit: activation === 'silu' ? swigluLimit : null,
  });

  try {
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

    const { workgroupsX, workgroupsY } = calculateFFNDispatch(variant, batchSize, intermediateSize);
    kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }

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
