
import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { shouldDequantizeToF16 } from './dtype-utils.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

// ============================================================================
// Norm Weight Offset Detection
// ============================================================================

export function needsNormWeightOffset(manifest) {
  if (!manifest) {
    debugTrace.loader('_needsNormWeightOffset: no manifest');
    return false;
  }

  const inferenceFlag = manifest.inference?.normalization?.rmsNormWeightOffset;
  if (inferenceFlag == null) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(
      `Manifest "${modelId}" is missing inference.normalization.rmsNormWeightOffset. ` +
      'Re-convert the model with a complete manifest.inference config.'
    );
  }

  if (inferenceFlag) {
    debugTrace.loader('RMSNorm weight offset enabled (manifest.inference.normalization.rmsNormWeightOffset=true)');
  }
  return inferenceFlag;
}

// ============================================================================
// Large Weight Handling
// ============================================================================

export function getLargeWeightConfig() {
  const config = getRuntimeConfig().inference.largeWeights;
  if (!config) {
    throw new Error('runtime.inference.largeWeights is required');
  }
  return config;
}

export function getLargeWeightMaxBytes() {
  const config = getLargeWeightConfig();
  if (!config.enabled) return null;

  const device = getDevice();
  if (!device) return null;

  const safety = Math.min(Math.max(config.safetyRatio, 0.1), 1);
  const maxBinding = Math.min(
    device.limits.maxStorageBufferBindingSize,
    device.limits.maxBufferSize
  );
  return Math.floor(maxBinding * safety);
}

export function estimateMatmulWeightBytes(location, gpuCapabilities, keepF32Weights) {
  if (!location.shape || location.shape.length === 0) return null;

  const numElements = location.shape.reduce((a, b) => a * b, 1);
  if (!Number.isFinite(numElements) || numElements <= 0) return null;

  const caps = gpuCapabilities || getKernelCapabilities();
  const hasF16 = caps?.hasF16 ?? false;
  const isMatmulWeight = shouldDequantizeToF16(location);

  const dtype = selectRuleValue('loader', 'weights', 'matmulWeightDtype', {
    locationDtype: location.dtype,
    hasF16,
    isMatmulWeight,
    keepF32Weights: Boolean(keepF32Weights),
  });

  const bytesPerElement = DTYPE_SIZES[selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype })];
  return { bytes: numElements * bytesPerElement, dtype };
}

export function resolveWeightLayout(location) {
  const isEmbedding = location.role === 'embedding' || location.role === 'lm_head';
  const useColumnWise = isEmbedding && location.shape?.length === 2
    ? location.shape[0] < location.shape[1]
    : false;

  return selectRuleValue('loader', 'weights', 'weightLayout', {
    layout: location.layout ?? null,
    useColumnWise,
  });
}

export function shouldStreamLargeWeight(name, location, label, gpuCapabilities, keepF32Weights) {
  const maxBytes = getLargeWeightMaxBytes();
  if (!maxBytes) return false;

  const estimate = estimateMatmulWeightBytes(location, gpuCapabilities, keepF32Weights);
  if (!estimate) return false;

  if (estimate.bytes <= maxBytes) return false;

  // Check if dtype can be streamed (only float types)
  const canStream = location.dtype === 'F16' || location.dtype === 'F32' || location.dtype === 'BF16';
  if (!canStream) {
    log.warn(
      'Loader',
      `${label} weight "${name}" (${formatBytes(estimate.bytes)}) exceeds GPU binding limit (${formatBytes(maxBytes)}) ` +
      `but dtype ${location.dtype} cannot be streamed. Regenerate with F16/F32 weights.`
    );
    return false;
  }

  log.warn(
    'Loader',
    `${label} weight "${name}" (${formatBytes(estimate.bytes)}) exceeds GPU binding limit (${formatBytes(maxBytes)}). ` +
    'Using CPU-backed streaming.'
  );
  return true;
}

// ============================================================================
// MoE Detection
// ============================================================================

export function isMoEModel(manifest) {
  if (!manifest) return false;

  // Explicit MoE config
  return (manifest.moeConfig?.numExperts ?? 0) > 1;
}
