/**
 * Manifest Config - Model configuration resolution from manifest.
 *
 * Pure functions for extracting configuration from manifests:
 * - Q4K strategy (fused vs dequant)
 * - Norm weight offset detection (Gemma models)
 * - Large weight handling configuration
 * - Weight layout resolution
 *
 * @module loader/manifest-config
 */

import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { TensorLocation, ModelConfig, KernelCapabilities } from './loader-types.js';
import type { WeightLayout, WeightDtype } from '../gpu/weight-buffer.js';
import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { getActiveKernelPath, getActiveKernelPathSource, isActiveKernelPathFusedQ4K } from '../config/kernel-path-loader.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { shouldDequantizeToF16, isEmbeddingWeight } from './dtype-utils.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Q4K Strategy Configuration
// ============================================================================

export interface Q4KConfig {
  /** Use fused Q4K matmul kernels (4x memory savings) */
  useFusedQ4K: boolean;
  /** Q4K layout from manifest */
  q4kLayout: 'flat' | 'row_wise' | 'column_wise' | null;
  /** Keep weights as F32 (disable F16 downcasting) */
  keepF32Weights: boolean;
}

/**
 * Configure Q4K strategy based on manifest and capabilities.
 *
 * Decision factors:
 * - Active kernel path (if set)
 * - Subgroup support (required for fused Q4K)
 * - Q4K layout in manifest (column_wise disables fused)
 * - Debug flag override
 *
 * @param manifest - Model manifest
 * @param gpuCapabilities - GPU kernel capabilities
 * @returns Q4K configuration
 */
export function configureQ4KStrategy(
  manifest: RDRRManifest | null,
  gpuCapabilities: KernelCapabilities | null
): Q4KConfig {
  const activeKernelPath = getActiveKernelPath();
  const pathSource = getActiveKernelPathSource();
  const q4kLayout = (manifest?.config as { q4kLayout?: string } | undefined)?.q4kLayout;

  // Default to fused Q4K when subgroups are available (4x memory savings)
  const caps = gpuCapabilities || getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  let useFused = activeKernelPath ? isActiveKernelPathFusedQ4K() : hasSubgroups;

  // Debug flag override
  if (typeof window !== 'undefined' && (window as unknown as { DOPPLER_DISABLE_FUSED_Q4K?: boolean }).DOPPLER_DISABLE_FUSED_Q4K) {
    useFused = false;
  }

  // Column-wise layout is incompatible with fused kernels
  if (q4kLayout === 'column_wise') {
    useFused = false;
  }

  const pathLabel = activeKernelPath?.id ?? 'auto';
  debugTrace.loader(`Q4K config: fused=${useFused}, kernelPath=${pathLabel}, source=${pathSource}, layout=${q4kLayout ?? 'default'}, subgroups=${hasSubgroups}`);

  return {
    useFusedQ4K: useFused,
    q4kLayout: (q4kLayout as 'flat' | 'row_wise' | 'column_wise') ?? null,
    keepF32Weights: false,
  };
}

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
 * @param manifest - Model manifest
 * @returns Whether norm weight offset is needed
 */
export function needsNormWeightOffset(manifest: RDRRManifest | null): boolean {
  if (!manifest) {
    debugTrace.loader('_needsNormWeightOffset: no manifest');
    return false;
  }

  // Explicit flag in manifest (preferred)
  const inferenceFlag = manifest.inference?.normalization?.rmsNormWeightOffset;
  if (inferenceFlag !== undefined) {
    if (inferenceFlag) {
      debugTrace.loader('Applying +1 norm weight offset (manifest.inference.normalization.rmsNormWeightOffset=true)');
    }
    return inferenceFlag;
  }

  // Legacy fallback: infer from model family
  const config = (manifest.config || {}) as ModelConfig;
  const arch = config.architectures?.[0] || (manifest.architecture as string) || '';
  const modelType = config.model_type || '';

  const isGemma2 = /gemma.*2|gemma2/i.test(arch) || /gemma.*2|gemma2/i.test(modelType);
  const isGemma3 = /gemma.*3|gemma3/i.test(arch) || /gemma.*3|gemma3/i.test(modelType);
  const needsOffset = isGemma2 || isGemma3;

  if (needsOffset) {
    const family = isGemma2 ? 'Gemma 2' : 'Gemma 3';
    debugTrace.loader(`Applying +1 norm weight offset for ${family} layer norms (legacy fallback)`);
  }

  return needsOffset;
}

// ============================================================================
// Large Weight Handling
// ============================================================================

export interface LargeWeightConfig {
  enabled: boolean;
  safetyRatio: number;
  preferF16: boolean;
}

/**
 * Get large weight handling configuration from runtime config.
 */
export function getLargeWeightConfig(): LargeWeightConfig {
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
 * @returns Max bytes, or null if large weight handling is disabled
 */
export function getLargeWeightMaxBytes(): number | null {
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
 * @param name - Tensor name
 * @param location - Tensor location info
 * @param gpuCapabilities - GPU capabilities
 * @param keepF32Weights - Whether to keep F32 (skip F16 downcast)
 * @returns Estimated bytes and output dtype, or null if cannot estimate
 */
export function estimateMatmulWeightBytes(
  name: string,
  location: TensorLocation,
  gpuCapabilities: KernelCapabilities | null,
  keepF32Weights: boolean
): { bytes: number; dtype: WeightDtype } | null {
  if (!location.shape || location.shape.length === 0) return null;

  const numElements = location.shape.reduce((a, b) => a * b, 1);
  if (!Number.isFinite(numElements) || numElements <= 0) return null;

  const caps = gpuCapabilities || getKernelCapabilities();
  const hasF16 = caps?.hasF16 ?? false;
  const isMatmulWeight = shouldDequantizeToF16(name);

  let dtype: WeightDtype = 'f32';
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
 * @param location - Tensor location info
 * @param name - Tensor name
 * @returns Weight layout ('row' or 'column')
 */
export function resolveWeightLayout(location: TensorLocation, name: string): WeightLayout {
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
 * @param name - Tensor name
 * @param location - Tensor location info
 * @param label - Human-readable label for logging (e.g., 'Embedding', 'LM head')
 * @param gpuCapabilities - GPU capabilities
 * @param keepF32Weights - Whether to keep F32
 * @returns Whether to use streaming
 */
export function shouldStreamLargeWeight(
  name: string,
  location: TensorLocation,
  label: string,
  gpuCapabilities: KernelCapabilities | null,
  keepF32Weights: boolean
): boolean {
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
 * @param manifest - Model manifest
 * @returns Whether model is MoE
 */
export function isMoEModel(manifest: RDRRManifest | null): boolean {
  if (!manifest) return false;

  // Explicit MoE config
  if (manifest.moeConfig != null) return true;

  // Check num_local_experts in config
  const config = manifest.config as ModelConfig | undefined;
  if ((config?.num_local_experts ?? 0) > 1) return true;

  return false;
}
