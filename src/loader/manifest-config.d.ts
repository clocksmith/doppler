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
import type { TensorLocation, KernelCapabilities } from './loader-types.js';
import type { WeightLayout, WeightDtype } from '../gpu/weight-buffer.js';

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
export declare function configureQ4KStrategy(
  manifest: RDRRManifest | null,
  gpuCapabilities: KernelCapabilities | null
): Q4KConfig;

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
export declare function needsNormWeightOffset(manifest: RDRRManifest | null): boolean;

export interface LargeWeightConfig {
  enabled: boolean;
  safetyRatio: number;
  preferF16: boolean;
}

/**
 * Get large weight handling configuration from runtime config.
 */
export declare function getLargeWeightConfig(): LargeWeightConfig;

/**
 * Get maximum bytes for a single GPU buffer binding.
 *
 * @returns Max bytes, or null if large weight handling is disabled
 */
export declare function getLargeWeightMaxBytes(): number | null;

/**
 * Estimate GPU memory required for a matmul weight after dequantization.
 *
 * @param name - Tensor name
 * @param location - Tensor location info
 * @param gpuCapabilities - GPU capabilities
 * @param keepF32Weights - Whether to keep F32 (skip F16 downcast)
 * @returns Estimated bytes and output dtype, or null if cannot estimate
 */
export declare function estimateMatmulWeightBytes(
  name: string,
  location: TensorLocation,
  gpuCapabilities: KernelCapabilities | null,
  keepF32Weights: boolean
): { bytes: number; dtype: WeightDtype } | null;

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
export declare function resolveWeightLayout(location: TensorLocation, name: string): WeightLayout;

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
export declare function shouldStreamLargeWeight(
  name: string,
  location: TensorLocation,
  label: string,
  gpuCapabilities: KernelCapabilities | null,
  keepF32Weights: boolean
): boolean;

/**
 * Check if model uses Mixture of Experts architecture.
 *
 * @param manifest - Model manifest
 * @returns Whether model is MoE
 */
export declare function isMoEModel(manifest: RDRRManifest | null): boolean;
