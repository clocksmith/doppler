/**
 * Manifest Config - Model configuration resolution from manifest.
 *
 * Pure functions for extracting configuration from manifests:
 * - Norm weight offset detection (Gemma models)
 * - Large weight handling configuration
 * - Weight layout resolution
 *
 * @module loader/manifest-config
 */

import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { shouldDequantizeToF16, isEmbeddingWeight } from './dtype-utils.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Norm Weight Offset Detection
// ============================================================================

/**
 * Check if model requires (1 + weight) offset for RMSNorm weights.
 *
 * GGUF files do NOT have the offset baked in - they store raw weights.
 * The +1 offset is applied at load time based on the manifest's config flag.
 *
 * Supported detection methods (in priority order):
 * 1. manifest.inference.normalization.rmsNormWeightOffset (explicit)
 * 2. Model family detection from architecture string (legacy fallback)
 *
 * @param {import('../storage/rdrr-format.js').RDRRManifest | null} manifest - Model manifest
 * @returns {boolean} Whether norm weight offset is needed
 */
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
    debugTrace.loader('Applying +1 norm weight offset (manifest.inference.normalization.rmsNormWeightOffset=true)');
  }
  return inferenceFlag;
}

// ============================================================================
// Large Weight Handling
// ============================================================================

/**
 * Get large weight handling configuration from runtime config.
 * @returns {import('./manifest-config.js').LargeWeightConfig}
 */
export function getLargeWeightConfig() {
  const config = getRuntimeConfig().inference.largeWeights;
  return {
    enabled: config?.enabled ?? false,
    safetyRatio: config?.safetyRatio ?? 0.9,
    preferF16: config?.preferF16 ?? true,
  };
}

/**
 * Get maximum bytes for a single GPU buffer binding.
 *
 * @returns {number | null} Max bytes, or null if large weight handling is disabled
 */
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

/**
 * Estimate GPU memory required for a matmul weight after dequantization.
 *
 * @param {string} name - Tensor name
 * @param {import('./loader-types.js').TensorLocation} location - Tensor location info
 * @param {import('./loader-types.js').KernelCapabilities | null} gpuCapabilities - GPU capabilities
 * @param {boolean} keepF32Weights - Whether to keep F32 (skip F16 downcast)
 * @returns {{ bytes: number; dtype: import('../gpu/weight-buffer.js').WeightDtype } | null} Estimated bytes and output dtype, or null if cannot estimate
 */
export function estimateMatmulWeightBytes(name, location, gpuCapabilities, keepF32Weights) {
  if (!location.shape || location.shape.length === 0) return null;

  const numElements = location.shape.reduce((a, b) => a * b, 1);
  if (!Number.isFinite(numElements) || numElements <= 0) return null;

  const caps = gpuCapabilities || getKernelCapabilities();
  const hasF16 = caps?.hasF16 ?? false;
  const isMatmulWeight = shouldDequantizeToF16(name);

  /** @type {import('../gpu/weight-buffer.js').WeightDtype} */
  let dtype = 'f32';
  switch (location.dtype) {
    case 'F16':
      dtype = 'f16';
      break;
    case 'BF16':
      dtype = hasF16 && isMatmulWeight ? 'f16' : 'f32';
      break;
    case 'Q4_K':
    case 'Q4_K_M':
      dtype = (hasF16 && isMatmulWeight && !keepF32Weights) ? 'f16' : 'f32';
      break;
    case 'Q6_K':
      dtype = 'f16';
      break;
    default:
      dtype = 'f32';
      break;
  }

  const bytesPerElement = DTYPE_SIZES[dtype === 'f16' ? 'f16' : 'f32'];
  return { bytes: numElements * bytesPerElement, dtype };
}

/**
 * Resolve weight layout from tensor location and name.
 *
 * Column layout is used for:
 * - Explicit layout='column' in tensor info
 * - Embeddings with transposed shape (dim0 < dim1)
 *
 * @param {import('./loader-types.js').TensorLocation} location - Tensor location info
 * @param {string} name - Tensor name
 * @returns {import('../gpu/weight-buffer.js').WeightLayout} Weight layout ('row' or 'column')
 */
export function resolveWeightLayout(location, name) {
  // Explicit layout from manifest
  if (location.layout === 'column') return 'column';

  // Embeddings may be transposed
  if (isEmbeddingWeight(name) && location.shape?.length === 2) {
    const [dim0, dim1] = location.shape;
    if (dim0 < dim1) {
      return 'column';
    }
  }

  return 'row';
}

/**
 * Check if a large weight should use CPU streaming instead of GPU buffer.
 *
 * Logs a warning if the weight exceeds GPU limits and provides guidance.
 *
 * @param {string} name - Tensor name
 * @param {import('./loader-types.js').TensorLocation} location - Tensor location info
 * @param {string} label - Human-readable label for logging (e.g., 'Embedding', 'LM head')
 * @param {import('./loader-types.js').KernelCapabilities | null} gpuCapabilities - GPU capabilities
 * @param {boolean} keepF32Weights - Whether to keep F32
 * @returns {boolean} Whether to use streaming
 */
export function shouldStreamLargeWeight(name, location, label, gpuCapabilities, keepF32Weights) {
  const maxBytes = getLargeWeightMaxBytes();
  if (!maxBytes) return false;

  const estimate = estimateMatmulWeightBytes(name, location, gpuCapabilities, keepF32Weights);
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

/**
 * Check if model uses Mixture of Experts architecture.
 *
 * @param {import('../storage/rdrr-format.js').RDRRManifest | null} manifest - Model manifest
 * @returns {boolean} Whether model is MoE
 */
export function isMoEModel(manifest) {
  if (!manifest) return false;

  // Explicit MoE config
  if (manifest.moeConfig != null) return true;

  // Check num_local_experts in config
  const config = /** @type {import('./loader-types.js').ModelConfig | undefined} */ (manifest.config);
  if ((config?.num_local_experts ?? 0) > 1) return true;

  return false;
}
