

import { getDevice } from '../device.js';
import { acquireBuffer, getBufferRequestedSize } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { WORKGROUP_SIZES } from './constants.js';
import { getKernelThresholds } from '../../config/schema/kernel-thresholds.schema.js';
import { trace } from '../../debug/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';


function resolveNormWeightDtype(byteSize, hiddenSize) {
  if (!byteSize || hiddenSize == null) return null;
  const f16Bytes = hiddenSize * 2;
  const f32Bytes = hiddenSize * 4;
  const sizeMatchesF32 = byteSize >= f32Bytes;
  const sizeMatchesF16 = byteSize >= f16Bytes && !sizeMatchesF32;
  if (!sizeMatchesF16 && !sizeMatchesF32) {
    return null;
  }
  return selectLoaderRule('loader', 'weights', 'normWeightDtypeFromSize', {
    sizeMatchesF16,
    sizeMatchesF32,
  });
}


export function selectMatmulRMSNormFusedVariant(N, dtype) {
  if (!dtype) {
    throw new Error('[MatmulRMSNormFused] dtype is required for variant selection.');
  }
  const isF16 = dtype === 'f16';
  const isSmall = N <= WORKGROUP_SIZES.DEFAULT;
  return selectRuleValue('fusedMatmulRmsnorm', 'variant', { isSmall, isF16 });
}


export async function runMatmulRMSNormFused(
  input,
  weight,
  normWeight,
  options
) {
  const device = getDevice();
  const {
    N,
    K,
    eps,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
    rmsNormWeightOffset = false,
  } = options;
  if (eps == null) {
    throw new Error('[MatmulRMSNormFused] eps is required.');
  }

  const { maxMediumN } = getKernelThresholds().fusedMatmul;
  if (N > maxMediumN) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds maxMediumN=${maxMediumN}; kernel only supports single-workgroup RMSNorm.`);
  }

  const weightBuffer = getBuffer(weight);
  const normWeightBuffer = getBuffer(normWeight);
  const normWeightSize = getBufferRequestedSize(normWeightBuffer);
  const normWeightDtype = resolveNormWeightDtype(normWeightSize, N);
  if (!normWeightDtype) {
    throw new Error(
      `[MatmulRMSNormFused] norm weight size (${normWeightSize} bytes) does not match ` +
      `hiddenSize=${N} (expected ${N * 2} or ${N * 4} bytes).`
    );
  }

  // Select variant based on output size and input dtype
  if (!input.dtype) {
    throw new Error('[MatmulRMSNormFused] input dtype is required.');
  }
  const dtype = input.dtype;
  const variant = selectMatmulRMSNormFusedVariant(N, dtype);

  trace.kernels(`MatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, dtype=${dtype}, hasResidual=${!!residual}, transposeB=${transposeB}, offset=${rmsNormWeightOffset}`);

  const constants = { RMS_NORM_OFFSET: rmsNormWeightOffset, WEIGHT_IS_F16: normWeightDtype === 'f16' };
  const pipeline = await getPipelineFast('fused_matmul_rmsnorm', variant, null, constants);

  // Output buffer: [1, N] - size depends on dtype
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const outputSize = N * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Create uniform buffer (8 u32/f32 = 32 bytes, padded for alignment)
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
      // Padding bytes 20-31 are zero-initialized
    },
    null,
    device
  );

  // Create placeholder for residual if not provided
  const residualBuffer = residual || device.createBuffer({
    label: 'matmul_rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_rmsnorm_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: normWeightBuffer } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } },
    ],
  });

  // Calculate workgroups
  
  const workgroups = 1;

  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Cleanup
  uniformBuffer.destroy();
  if (!residual) residualBuffer.destroy();

  // Output dtype matches input dtype
  return createTensor(output, input.dtype, [1, N], 'matmul_rmsnorm_fused_output');
}


export async function recordMatmulRMSNormFused(
  recorder,
  input,
  weight,
  normWeight,
  options
) {
  const device = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
    rmsNormWeightOffset = false,
  } = options;

  const { maxMediumN } = getKernelThresholds().fusedMatmul;
  if (N > maxMediumN) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds maxMediumN=${maxMediumN}; kernel only supports single-workgroup RMSNorm.`);
  }

  const weightBuffer = getBuffer(weight);
  const normWeightBuffer = getBuffer(normWeight);
  const normWeightSize = getBufferRequestedSize(normWeightBuffer);
  const normWeightDtype = resolveNormWeightDtype(normWeightSize, N);
  if (!normWeightDtype) {
    throw new Error(
      `[MatmulRMSNormFused] norm weight size (${normWeightSize} bytes) does not match ` +
      `hiddenSize=${N} (expected ${N * 2} or ${N * 4} bytes).`
    );
  }

  // Select variant based on dtype
  if (!input.dtype) {
    throw new Error('[MatmulRMSNormFused] input dtype is required.');
  }
  const dtype = input.dtype;
  const variant = selectMatmulRMSNormFusedVariant(N, dtype);

  trace.kernels(`recordMatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, dtype=${dtype}, hasResidual=${!!residual}, transposeB=${transposeB}, offset=${rmsNormWeightOffset}`);

  const constants = { RMS_NORM_OFFSET: rmsNormWeightOffset, WEIGHT_IS_F16: normWeightDtype === 'f16' };
  const pipeline = await getPipelineFast('fused_matmul_rmsnorm', variant, null, constants);

  // Output buffer - size depends on dtype
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const outputSize = N * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Uniform buffer via recorder (8 u32/f32 = 32 bytes, padded for alignment)
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
      // Padding bytes 20-31 are zero-initialized
    },
    recorder
  );

  // Placeholder for residual
  const residualBuffer = residual || device.createBuffer({
    label: 'matmul_rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_rmsnorm_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: normWeightBuffer } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } },
    ],
  });

  // Calculate workgroups
  
  const workgroups = 1;

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Track placeholder for cleanup
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }

  // Output dtype matches input dtype
  return createTensor(output, input.dtype, [1, N], 'matmul_rmsnorm_fused_output');
}


export function shouldUseFusedMatmulRMSNorm(M, N, K) {
  // Only beneficial for decode (M=1)
  if (M !== 1) {
    return false;
  }

  const { maxMediumN, maxMediumK } = getKernelThresholds().fusedMatmul;
  if (N > maxMediumN) {
    return false;
  }

  if (typeof K === 'number' && K > maxMediumK) {
    return false;
  }

  return true;
}
