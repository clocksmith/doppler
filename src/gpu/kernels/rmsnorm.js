

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer, getBufferRequestedSize } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';


function canUseF16(input, residual) {
  if (input.dtype !== 'f16') return false;
  if (residual && residual.dtype !== 'f16') return false;
  return true;
}


function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}


function resolveNormWeightDtype(options, hiddenSize) {
  // If explicit dtype provided, use rule to validate/select
  if (options.normWeightDtype) {
    return selectLoaderRule('loader', 'weights', 'normWeightDtype', {
      normWeightDtype: options.normWeightDtype,
    });
  }
  // Fallback: infer from buffer size (tolerant of bucketing)
  // Use getBufferRequestedSize to get unbucketed size when tracked
  const weight = options._weightBuffer;
  if (!weight || hiddenSize == null) return 'f32';
  const byteSize = getBufferRequestedSize(weight);
  const f16Bytes = hiddenSize * 2;
  const f32Bytes = hiddenSize * 4;
  // Tolerant matching: check if size can hold expected elements
  // F16: needs at least hiddenSize*2 bytes, less than hiddenSize*4
  // F32: needs at least hiddenSize*4 bytes
  return selectLoaderRule('shared', 'dtype', 'dtypeFromSize', {
    bytesPerElement: byteSize >= f32Bytes ? 4 : byteSize >= f16Bytes ? 2 : 4,
  });
}


export function selectRMSNormKernel(options = {}, isF16 = false) {
  const { residual = null, hiddenSize = null } = options;
  const { smallThreshold } = getKernelThresholds().rmsnorm;

  // Check if subgroups are available
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const isSmall = hiddenSize !== null && hiddenSize <= smallThreshold;
  return selectRuleValue(
    'rmsnorm',
    'variant',
    { isF16, residual: !!residual, hasSubgroups, isSmall }
  );
}


export async function runRMSNorm(
  input,
  weight,
  eps = 1e-5,
  options = {}
) {
  const device = getDevice();
  const { batchSize = 1, hiddenSize, residual = null, outputBuffer = null, rmsNormWeightOffset = false } = options;

  // Check if F16 can be used based on tensor dtypes
  const isF16 = canUseF16(input, residual);
  const variant = selectRMSNormKernel(options, isF16);
  trace.kernels(`RMSNorm: input.dtype=${input.dtype}, isF16=${isF16}, variant=${variant}, offset=${rmsNormWeightOffset}`);

  if (residual) {
    trace.kernels(`RMSNorm: Using residual variant, residual.size=${residual.buffer.size}, inferredHiddenSize=${hiddenSize ?? 'auto'}, batchSize=${batchSize}`);
  }

  // Define constants for the pipeline
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  if (inferredHiddenSize == null) {
    throw new Error('RMSNorm requires hiddenSize or input shape to infer it.');
  }
  // Resolve weight dtype via rule (explicit option or inferred from buffer)
  const weightDtype = resolveNormWeightDtype({ ...options, _weightBuffer: weight }, inferredHiddenSize);

  const constants = {
    RMS_NORM_OFFSET: rmsNormWeightOffset,
    WEIGHT_IS_F16: weightDtype === 'f16',
  };

  const pipeline = await getPipelineFast('rmsnorm', variant, null, constants);

  // Create output buffer if not provided
  const bytesPerElement = isF16 ? 2 : 4;
  const outputSize = batchSize * inferredHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

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
  if (hasResidualFlag) {
    trace.kernels(`RMSNorm: Uniform hasResidual=${hasResidualFlag}, hiddenSize=${inferredHiddenSize}, batchSize=${batchSize}`);
  }

  // Shader expects 5 bindings - create placeholder when no residual (uniform flags it as unused)
  const residualBuffer = residual?.buffer || device.createBuffer({
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: residualBuffer } },
    ],
  });

  dispatch(device, pipeline, bindGroup, batchSize, 'rmsnorm');

  uniformBuffer.destroy();
  if (!residual) residualBuffer.destroy();

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
}


export async function recordRMSNorm(
  recorder,
  input,
  weight,
  eps = 1e-5,
  options = {}
) {
  const device = recorder.device;
  const {
    batchSize = 1,
    hiddenSize = null,
    residual = null,
    outputBuffer = null,
    rmsNormWeightOffset = false,
  } = options;

  // Check if F16 can be used based on tensor dtypes
  const isF16 = canUseF16(input, residual);
  const variant = selectRMSNormKernel(options, isF16);
  const bytesPerElement = isF16 ? 2 : 4;

  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  if (inferredHiddenSize == null) {
    throw new Error('RMSNorm requires hiddenSize or input shape to infer it.');
  }
  // Resolve weight dtype via rule (explicit option or inferred from buffer)
  const weightDtype = resolveNormWeightDtype({ ...options, _weightBuffer: weight }, inferredHiddenSize);
  const outputSize = batchSize * inferredHiddenSize * bytesPerElement;

  // Define constants for the pipeline
  const constants = {
    RMS_NORM_OFFSET: rmsNormWeightOffset,
    WEIGHT_IS_F16: weightDtype === 'f16',
  };

  const pipeline = await getPipelineFast('rmsnorm', variant, null, constants);

  // Output buffer
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

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
  const residualBuffer = residual?.buffer || device.createBuffer({
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: residualBuffer } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, batchSize, 'rmsnorm');

  // Track dummy buffer for cleanup if we created it
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
}
