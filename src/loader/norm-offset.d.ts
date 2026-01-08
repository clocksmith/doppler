/**
 * Norm Offset - Apply +1 offset to RMSNorm weights.
 *
 * Gemma 3+ models use (1 + weight) formula for RMSNorm instead of
 * just weight. This module handles the transformation.
 *
 * @module loader/norm-offset
 */

export interface NormOffsetOptions {
  /** Actual number of elements (avoids reading buffer pool padding) */
  actualNumElements?: number;
  /** Buffer dtype for conversion */
  bufferDtype?: 'f16' | 'f32' | 'bf16';
  /** Enable debug logging for first transformation */
  enableDebugLog?: boolean;
}

export interface NormOffsetResult {
  /** The transformed tensor */
  tensor: GPUBuffer | Float32Array;
  /** Whether debug logging was performed */
  debugLogged: boolean;
}

/**
 * Apply +1 offset to norm weights for Gemma 3+ models.
 *
 * Transforms weight values from `w` to `1 + w`.
 *
 * IMPORTANT: actualNumElements must be provided to avoid reading garbage
 * padding from the buffer pool's power-of-2 bucketing.
 *
 * @param tensor - Input tensor (GPUBuffer or Float32Array)
 * @param options - Offset options
 * @returns Transformed tensor result
 */
export declare function applyNormWeightOffset(
  tensor: GPUBuffer | Float32Array,
  options?: NormOffsetOptions
): Promise<NormOffsetResult>;
