

import { getKernelCapabilities } from '../device.js';
import { acquireBuffer, getBufferRequestedSize, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';
import { getBuffer, getWeightDtype, getBufferDtype } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';

// Conservative fallback dtype for norm weight inference when metadata is unavailable.
const DEFAULT_DTYPE = 'f32';

// Must equal MAX_CACHE_SIZE in rmsnorm.wgsl. The "residual" variant routes to
// main_cached, which caches the input in shared_cache[MAX_CACHE_SIZE]. When
// hiddenSize exceeds this limit, OOB cache writes silently drop indices and the
// second pass reads zero for those dims, producing wrong output (only residual
// preserved). Models with hiddenSize > limit must skip the cached variant.
export const RMSNORM_CACHE_LIMIT = 4608;

export function residualVariantBypassesCache(residual, hiddenSize) {
  return !!residual && hiddenSize !== null && hiddenSize !== undefined && hiddenSize > RMSNORM_CACHE_LIMIT;
}

function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}

function normalizeNormWeightDtype(dtype) {
  if (typeof dtype !== 'string') return null;
  const value = dtype.toLowerCase();
  if (value === 'f16' || value === 'f32') {
    return value;
  }
  return null;
}

function resolveNormWeightDtype(weight, hiddenSize) {
  const explicitDtype = normalizeNormWeightDtype(getWeightDtype(weight));
  if (explicitDtype) {
    return explicitDtype;
  }

  const weightBuffer = getBuffer(weight);
  const taggedDtype = normalizeNormWeightDtype(getBufferDtype(weightBuffer));
  if (taggedDtype) {
    return taggedDtype;
  }

  // Conservative fallback: f32 avoids precision loss when dtype cannot be determined.
  // This path fires for non-GPU buffers or missing hiddenSize, both of which prevent
  // size-based dtype inference below.
  const hasGPUBufferType = typeof GPUBuffer !== 'undefined';
  if (!hasGPUBufferType || !(weightBuffer instanceof GPUBuffer) || hiddenSize == null || hiddenSize <= 0) {
    return DEFAULT_DTYPE;
  }

  const byteSize = getBufferRequestedSize(weightBuffer);
  const f16Bytes = hiddenSize * 2;
  const f32Bytes = hiddenSize * 4;
  const sizeMatchesF16 = byteSize === f16Bytes;
  const sizeMatchesF32 = byteSize === f32Bytes;
  if (sizeMatchesF16 || sizeMatchesF32) {
    return selectLoaderRule('loader', 'weights', 'normWeightDtypeFromSize', {
      sizeMatchesF16,
      sizeMatchesF32,
    });
  }
  // Buffer size matches neither f16 nor f32 for given hiddenSize; fall back to f32.
  return DEFAULT_DTYPE;
}

function assertRMSNormWeightBuffer(weight, weightBuffer, hiddenSize) {
  const isGpuBuffer = weightBuffer && (
    typeof GPUBuffer === 'undefined'
      ? true
      : weightBuffer instanceof GPUBuffer
  );
  if (isGpuBuffer) {
    return;
  }
  const weightLabel = weight?.label ?? 'unknown';
  const weightType = weight === null ? 'null' : weight === undefined ? 'undefined' : weight.constructor?.name || typeof weight;
  const bufferType = weightBuffer === null ? 'null' : weightBuffer === undefined ? 'undefined' : weightBuffer.constructor?.name || typeof weightBuffer;
  throw new Error(
    `[rmsnorm] weight "${weightLabel}" requires a GPUBuffer ` +
    `(weightType=${weightType}, bufferType=${bufferType}, hiddenSize=${hiddenSize ?? 'unknown'}).`
  );
}

function planRMSNormDispatch(target, numTokens) {
  const device = target?.device;
  const maxPerDim = Number.isFinite(device?.limits?.maxComputeWorkgroupsPerDimension)
    ? device.limits.maxComputeWorkgroupsPerDimension
    : 65535;
  const tokenStride = Math.min(numTokens, maxPerDim);
  return {
    tokenStride,
    workgroups: [tokenStride, Math.ceil(numTokens / tokenStride), 1],
  };
}

export function selectRMSNormKernel(options = {}, isF16 = false) {
  const { residual = null, hiddenSize = null } = options;
  const { smallThreshold } = getKernelThresholds().rmsnorm;
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const isSmall = hiddenSize !== null && hiddenSize <= smallThreshold;
  return selectRuleValue(
    'rmsnorm',
    'variant',
    {
      isF16,
      residual: !!residual && !residualVariantBypassesCache(residual, hiddenSize),
      hasSubgroups,
      isSmall,
    }
  );
}

export async function runRMSNorm(
  input,
  weight,
  eps,
  options = {}
) {
  const {
    batchSize = 1, hiddenSize, residual = null, outputBuffer = null,
    rmsNormWeightOffset = false, preResidual = null, residualSumOutput = null,
  } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectRMSNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  const normWeightBuffer = getBuffer(weight);
  assertRMSNormWeightBuffer(weight, normWeightBuffer, inferredHiddenSize);
  const normWeightDtype = resolveNormWeightDtype(weight, inferredHiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');
  const ownedOutput = outputBuffer ? null : outputBuf;
  const dispatchPlan = planRMSNormDispatch(null, batchSize);

  // Shader layout always includes the residual binding; when unused, bind a harmless placeholder.
  const effectiveResidual = preResidual || residual;
  const residualBuf = effectiveResidual?.buffer || effectiveResidual || input?.buffer || input || outputBuf;
  const hasPrenormOutput = !!preResidual && !!residualSumOutput;
  const kernelBindings = [input, normWeightBuffer, outputBuf, residualBuf];
  // Binding 5 (residual_sum_output) must not alias binding 3 (output) — both are read_write.
  // Allocate a small placeholder when the prenorm output path is inactive.
  const ownedPrenormPlaceholder = hasPrenormOutput ? null : acquireBuffer(4, undefined, 'rmsnorm_prenorm_placeholder');
  const prenormBuf = hasPrenormOutput
    ? (residualSumOutput?.buffer || residualSumOutput)
    : ownedPrenormPlaceholder;
  const extraBindings = [{ binding: 5, buffer: prenormBuf }];

  try {
    await unifiedKernelWrapper(
      'rmsnorm',
      null,
      variant,
      kernelBindings,
      {
        hidden_size: inferredHiddenSize,
        num_tokens: batchSize,
        eps,
        has_residual: residual ? 1 : 0,
        token_stride: dispatchPlan.tokenStride,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        WEIGHT_IS_F16: normWeightDtype === 'f16',
        PRE_RESIDUAL: !!preResidual,
        OUTPUT_PRENORM: hasPrenormOutput,
      },
      extraBindings
    );

    if (ownedPrenormPlaceholder) releaseBuffer(ownedPrenormPlaceholder);

    return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
  } catch (error) {
    if (ownedPrenormPlaceholder) releaseBuffer(ownedPrenormPlaceholder);
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
}

export async function recordRMSNorm(
  recorder,
  input,
  weight,
  eps,
  options = {}
) {
  const {
    batchSize = 1, hiddenSize = null, residual = null, outputBuffer = null,
    rmsNormWeightOffset = false, preResidual = null, residualSumOutput = null,
  } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectRMSNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  const normWeightBuffer = getBuffer(weight);
  assertRMSNormWeightBuffer(weight, normWeightBuffer, inferredHiddenSize);
  const normWeightDtype = resolveNormWeightDtype(weight, inferredHiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');
  const ownedOutput = outputBuffer ? null : outputBuf;
  const dispatchPlan = planRMSNormDispatch(recorder, batchSize);

  const effectiveResidual = preResidual || residual;
  const residualBuf = effectiveResidual?.buffer || effectiveResidual || input?.buffer || input || outputBuf;
  const hasPrenormOutput = !!preResidual && !!residualSumOutput;
  const kernelBindings = [input, normWeightBuffer, outputBuf, residualBuf];
  const ownedPrenormPlaceholder = hasPrenormOutput ? null : acquireBuffer(4, undefined, 'rmsnorm_prenorm_placeholder');
  const prenormBuf = hasPrenormOutput
    ? (residualSumOutput?.buffer || residualSumOutput)
    : ownedPrenormPlaceholder;
  const extraBindings = [{ binding: 5, buffer: prenormBuf }];

  try {
    await unifiedKernelWrapper(
      'rmsnorm',
      recorder,
      variant,
      kernelBindings,
      {
        hidden_size: inferredHiddenSize,
        num_tokens: batchSize,
        eps,
        has_residual: residual ? 1 : 0,
        token_stride: dispatchPlan.tokenStride,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        WEIGHT_IS_F16: normWeightDtype === 'f16',
        PRE_RESIDUAL: !!preResidual,
        OUTPUT_PRENORM: hasPrenormOutput,
      },
      extraBindings
    );

    if (ownedPrenormPlaceholder) releaseBuffer(ownedPrenormPlaceholder);
    return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
  } catch (error) {
    if (ownedPrenormPlaceholder) releaseBuffer(ownedPrenormPlaceholder);
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
}
