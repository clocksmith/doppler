

import { getDevice, getDeviceEpoch, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { FFN_DISPATCH } from './constants.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { getPipelineBindGroupLayout } from './pipeline-cache.js';
import { recordDispatch } from './dispatch.js';
import { trace, isTraceEnabled } from '../../debug/index.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { isFusedQ4KDisabled } from './matmul.js';
import { getKernelThresholds, QK_K } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { resolveNormWeightDtype } from './rmsnorm.js';
import { getShaderModule } from './shader-cache.js';

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
  'f16_batched',
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

const Q4K_VARIANTS = new Set([
  'q4k',
  'q4k_metal_simd16',
  'q4k_batched',
  'q4k_f16a',
  'q4k_batched_f16a',
]);

function selectFFNVariant(batchSize, weightDtype, intermediateSize, hiddenSize, inputDtype, variantOverride = null) {
  const { multiOutputThreshold } = getKernelThresholds().ffn;
  const capabilities = getKernelCapabilities();
  const isQ4K = weightDtype === 'q4k';
  const fusedAllowed = !isFusedQ4KDisabled();
  const hiddenSubblockAligned = hiddenSize % 32 === 0;
  const useMultiOutput = intermediateSize <= multiOutputThreshold;
  const hasF16 = capabilities.hasF16;
  const useF16Input = inputDtype === 'f16';

  const selected = selectRuleValue(
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
  if (variantOverride == null) {
    return selected;
  }
  if (variantOverride !== 'q4k_metal_simd16') {
    throw new Error(`Fused FFN does not support explicit variant "${String(variantOverride)}".`);
  }
  if (selected !== 'q4k') {
    throw new Error(
      `Fused FFN variant "${variantOverride}" requires single-token Q4_K decode with f32 activations; ` +
      `the resolved base variant is "${selected}".`
    );
  }
  return variantOverride;
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

function createNormedFFNUniformBuffer(device, recorder, params) {
  const swigluLimit = resolveSwigluLimit(params.swigluLimit, 'FusedNormedFFN uniforms');
  return createUniformBufferWithView(
    'fused_normed_ffn_uniforms',
    48,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === 'silu' ? 0 : 1, true);
      view.setUint32(20, Math.ceil(params.hiddenSize / QK_K), true);
      view.setFloat32(24, swigluLimit, true);
      view.setUint32(28, params.rmsNormWeightOffset ? 1 : 0, true);
      view.setUint32(32, params.normWeightDtype === 'f16' ? 1 : 0, true);
    },
    recorder,
    device
  );
}


let fusedNormedFFNEpoch = -1;
let fusedNormedFFNPipelines = new Map();

function pipelineConstantsKey(constants) {
  if (!constants) return 'default';
  return Object.keys(constants)
    .sort()
    .map((key) => `${key}=${String(constants[key])}`)
    .join(';');
}

async function getFusedNormedFFNPipeline(device, constants = null) {
  const epoch = getDeviceEpoch();
  if (fusedNormedFFNEpoch !== epoch) {
    fusedNormedFFNEpoch = epoch;
    fusedNormedFFNPipelines = new Map();
  }
  const pipelineKey = pipelineConstantsKey(constants);
  const cached = fusedNormedFFNPipelines.get(pipelineKey);
  if (cached) return cached;
  const module = await getShaderModule(device, 'fused_normed_ffn_q4k.wgsl', 'fused_normed_ffn_q4k');
  const pipeline = device.createComputePipeline({
    label: pipelineKey === 'default'
      ? 'fused_normed_ffn_q4k_pipeline'
      : `fused_normed_ffn_q4k_pipeline_${pipelineKey}`,
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
      ...(constants ? { constants } : {}),
    },
  });
  fusedNormedFFNPipelines.set(pipelineKey, pipeline);
  return pipeline;
}

function resolveSwigluLimit(value, context) {
  if (value === undefined) {
    throw new Error(`${context} requires an explicit swigluLimit (null or number).`);
  }
  if (value == null) return 0;
  return value;
}

function assertPositiveIntegerConstant(value, name, context) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${context} requires ${name} to be a positive integer; got ${String(value)}.`);
  }
}

function normalizeQ4KPipelineConstants(constants, context) {
  if (constants == null) return null;
  if (typeof constants !== 'object') {
    throw new Error(`${context} pipelineConstants must be an object or null.`);
  }
  const colsPerWorkgroup = constants.COLS_PER_WG;
  const threadsPerCol = constants.THREADS_PER_COL ?? constants.THREADS_PER_COL_GEMV;
  const workgroupSize = constants.WORKGROUP_SIZE ?? 256;
  const useFullBlockFastPath = constants.USE_FULL_BLOCK_FAST_PATH;
  const hasWorkgroupConstants = colsPerWorkgroup !== undefined ||
    threadsPerCol !== undefined ||
    constants.WORKGROUP_SIZE !== undefined;
  if (!hasWorkgroupConstants && useFullBlockFastPath === undefined) {
    return null;
  }
  if (useFullBlockFastPath !== undefined && typeof useFullBlockFastPath !== 'boolean') {
    throw new Error(`${context} requires USE_FULL_BLOCK_FAST_PATH to be a boolean; got ${String(useFullBlockFastPath)}.`);
  }
  const normalized = {};
  if (hasWorkgroupConstants) {
    assertPositiveIntegerConstant(colsPerWorkgroup, 'COLS_PER_WG', context);
    assertPositiveIntegerConstant(threadsPerCol, 'THREADS_PER_COL', context);
    assertPositiveIntegerConstant(workgroupSize, 'WORKGROUP_SIZE', context);
    if (colsPerWorkgroup * threadsPerCol !== workgroupSize) {
      throw new Error(
        `${context} requires COLS_PER_WG * THREADS_PER_COL to equal WORKGROUP_SIZE; ` +
        `got ${colsPerWorkgroup} * ${threadsPerCol} != ${workgroupSize}.`
      );
    }
    if (constants.WORKGROUP_SIZE !== undefined) {
      normalized.WORKGROUP_SIZE = workgroupSize;
    }
    normalized.COLS_PER_WG = colsPerWorkgroup;
    normalized.THREADS_PER_COL = threadsPerCol;
  }
  if (useFullBlockFastPath !== undefined) {
    normalized.USE_FULL_BLOCK_FAST_PATH = useFullBlockFastPath ? 1 : 0;
  }
  return normalized;
}

function q4kColsPerWorkgroup(constants) {
  return constants?.COLS_PER_WG ?? FFN_DISPATCH.Q4K_COLS_PER_WG;
}

function calculateFFNDispatch(variant, batchSize, intermediateSize, constants = null) {
  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    workgroupsX = Math.ceil(intermediateSize / FFN_DISPATCH.MULTI_OUTPUTS_PER_WG);
  } else if (Q4K_VARIANTS.has(variant)) {
    workgroupsX = Math.ceil(intermediateSize / q4kColsPerWorkgroup(constants));
    workgroupsY = (variant === 'q4k_batched' || variant === 'q4k_batched_f16a') ? batchSize : 1;
  } else if (variant === 'batched' || variant === 'f16_batched' || variant === 'f16_native_batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  return { workgroupsX, workgroupsY };
}

function resolveFusedFFNPipelineConstants(variant, hiddenSize, pipelineConstants = null) {
  if (Q4K_VARIANTS.has(variant)) {
    return normalizeQ4KPipelineConstants(pipelineConstants, 'FusedFFN Q4K');
  }
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
    pipelineConstants = null,
    variant: variantOverride = null,
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
  const variant = selectFFNVariant(
    batchSize,
    gateDtype,
    intermediateSize,
    hiddenSize,
    input.dtype,
    variantOverride
  );
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
  const constants = resolveFusedFFNPipelineConstants(variant, hiddenSize, pipelineConstants);
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
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: getBuffer(W_gate) } },
        { binding: 3, resource: { buffer: getBuffer(W_up) } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const { workgroupsX, workgroupsY } = calculateFFNDispatch(variant, batchSize, intermediateSize, constants);
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
    pipelineConstants = null,
    variant: variantOverride = null,
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
  const variant = selectFFNVariant(
    batchSize,
    gateDtype,
    intermediateSize,
    hiddenSize,
    input.dtype,
    variantOverride
  );
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
  const constants = resolveFusedFFNPipelineConstants(variant, hiddenSize, pipelineConstants);
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
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: getBuffer(W_gate) } },
        { binding: 3, resource: { buffer: getBuffer(W_up) } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const { workgroupsX, workgroupsY } = calculateFFNDispatch(variant, batchSize, intermediateSize, constants);
    kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }

  return createTensor(output, outputDtype, [batchSize, intermediateSize], 'fused_ffn_output');
}

function validateFusedNormedFFNInputs(input, invRmsBuffer, normWeight, W_gate, W_up, hiddenSize, intermediateSize, options) {
  const batchSize = options.batchSize ?? 1;
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error(`Fused normed FFN batchSize must be a positive integer; got ${String(batchSize)}.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`Fused normed FFN hiddenSize must be a positive integer; got ${String(hiddenSize)}.`);
  }
  if (!Number.isInteger(intermediateSize) || intermediateSize <= 0) {
    throw new Error(`Fused normed FFN intermediateSize must be a positive integer; got ${String(intermediateSize)}.`);
  }
  if (input?.dtype !== 'f32') {
    throw new Error(`Fused normed FFN requires f32 prenorm input; got ${String(input?.dtype)}.`);
  }
  if (!invRmsBuffer) {
    throw new Error('Fused normed FFN requires an inverse-RMS buffer.');
  }
  if (!normWeight) {
    throw new Error('Fused normed FFN requires a norm weight buffer.');
  }
  const gateDtype = getWeightDtype(W_gate);
  const upDtype = getWeightDtype(W_up);
  if (gateDtype !== 'q4k' || upDtype !== 'q4k') {
    throw new Error(`Fused normed FFN requires Q4_K gate/up weights; got gate=${gateDtype}, up=${upDtype}.`);
  }
  if (hiddenSize % 32 !== 0) {
    throw new Error(`Fused normed FFN requires hiddenSize aligned to 32; got ${hiddenSize}.`);
  }
  if (FFN_DISPATCH.Q4K_COLS_PER_WG <= 0 || 256 % FFN_DISPATCH.Q4K_COLS_PER_WG !== 0) {
    throw new Error(`Fused normed FFN requires Q4K_COLS_PER_WG to divide 256; got ${FFN_DISPATCH.Q4K_COLS_PER_WG}.`);
  }
  const activation = options.activation ?? 'silu';
  if (activation !== 'silu' && activation !== 'gelu') {
    throw new Error(`Fused normed FFN activation must be silu|gelu; got ${String(activation)}.`);
  }
  return {
    batchSize,
    activation,
    normWeightDtype: resolveNormWeightDtype(normWeight, hiddenSize),
  };
}

function createFusedNormedBindGroup(device, pipeline, uniformBuffer, input, normWeight, invRmsBuffer, W_gate, W_up, output) {
  return device.createBindGroup({
    label: 'fused_normed_ffn_q4k_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(normWeight) } },
      { binding: 3, resource: { buffer: invRmsBuffer } },
      { binding: 4, resource: { buffer: getBuffer(W_gate) } },
      { binding: 5, resource: { buffer: getBuffer(W_up) } },
      { binding: 6, resource: { buffer: output } },
    ],
  });
}

function calculateFusedNormedDispatch(batchSize, intermediateSize, constants = null) {
  return {
    workgroupsX: Math.ceil(intermediateSize / q4kColsPerWorkgroup(constants)),
    workgroupsY: batchSize,
  };
}

export async function runFusedFFNFromRMSNormStats(
  input,
  invRmsBuffer,
  normWeight,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const device = getDevice();
  const {
    batchSize,
    activation,
    normWeightDtype,
  } = validateFusedNormedFFNInputs(input, invRmsBuffer, normWeight, W_gate, W_up, hiddenSize, intermediateSize, options);
  resolveSwigluLimit(options.swigluLimit, 'FusedNormedFFN');
  const constants = normalizeQ4KPipelineConstants(options.pipelineConstants, 'FusedNormedFFN Q4K');
  const outputSize = batchSize * intermediateSize * 4;
  const ownedOutput = options.outputBuffer ? null : acquireBuffer(outputSize, undefined, 'fused_normed_ffn_output');
  const output = options.outputBuffer || ownedOutput;
  const uniformBuffer = createNormedFFNUniformBuffer(device, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha: options.alpha ?? 1.0,
    activation,
    swigluLimit: activation === 'silu' ? options.swigluLimit : null,
    rmsNormWeightOffset: options.rmsNormWeightOffset === true,
    normWeightDtype,
  });
  try {
    const pipeline = await getFusedNormedFFNPipeline(device, constants);
    const bindGroup = createFusedNormedBindGroup(device, pipeline, uniformBuffer, input, normWeight, invRmsBuffer, W_gate, W_up, output);
    const { workgroupsX, workgroupsY } = calculateFusedNormedDispatch(batchSize, intermediateSize, constants);
    const encoder = device.createCommandEncoder({ label: 'fused_normed_ffn_q4k_encoder' });
    const pass = encoder.beginComputePass({ label: 'fused_normed_ffn_q4k' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
  } catch (error) {
    if (ownedOutput) releaseBuffer(ownedOutput);
    uniformBuffer.destroy();
    throw error;
  }
  uniformBuffer.destroy();
  return createTensor(output, 'f32', [batchSize, intermediateSize], 'fused_normed_ffn_output');
}

export async function recordFusedFFNFromRMSNormStats(
  recorder,
  input,
  invRmsBuffer,
  normWeight,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const {
    batchSize,
    activation,
    normWeightDtype,
  } = validateFusedNormedFFNInputs(input, invRmsBuffer, normWeight, W_gate, W_up, hiddenSize, intermediateSize, options);
  resolveSwigluLimit(options.swigluLimit, 'FusedNormedFFN');
  const constants = normalizeQ4KPipelineConstants(options.pipelineConstants, 'FusedNormedFFN Q4K');
  const outputSize = batchSize * intermediateSize * 4;
  const ownedOutput = options.outputBuffer ? null : acquireBuffer(outputSize, undefined, 'fused_normed_ffn_output');
  const output = options.outputBuffer || ownedOutput;
  const uniformBuffer = createNormedFFNUniformBuffer(recorder.device, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha: options.alpha ?? 1.0,
    activation,
    swigluLimit: activation === 'silu' ? options.swigluLimit : null,
    rmsNormWeightOffset: options.rmsNormWeightOffset === true,
    normWeightDtype,
  });
  try {
    const pipeline = await getFusedNormedFFNPipeline(recorder.device, constants);
    const bindGroup = createFusedNormedBindGroup(recorder.device, pipeline, uniformBuffer, input, normWeight, invRmsBuffer, W_gate, W_up, output);
    const { workgroupsX, workgroupsY } = calculateFusedNormedDispatch(batchSize, intermediateSize, constants);
    recordDispatch(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_normed_ffn_q4k');
  } catch (error) {
    if (ownedOutput) releaseBuffer(ownedOutput);
    throw error;
  }
  return createTensor(output, 'f32', [batchSize, intermediateSize], 'fused_normed_ffn_output');
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
