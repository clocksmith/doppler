/**
 * Quantization Defaults Config Schema
 *
 * Default quantization settings for the model converter.
 * Controls the target precision for different weight groups when not explicitly specified.
 *
 * @module config/schema/quantization-defaults
 */

// =============================================================================
// Quantization Dtype Types
// =============================================================================

/** Supported weight quantization types */
export type WeightQuantType = 'f16' | 'f32' | 'q4k' | 'q6k' | 'q8_0';

/** Supported embedding quantization types */
export type EmbeddingQuantType = 'f16' | 'f32';

// =============================================================================
// Quantization Defaults Schema
// =============================================================================

/**
 * Default quantization settings for model conversion.
 *
 * These defaults are used when no explicit quantization is specified
 * for a particular weight group during conversion.
 */
export interface QuantizationDefaultsSchema {
  /** Default dtype for vision encoder weights (default: 'f16') */
  visionDtype: EmbeddingQuantType;

  /** Default dtype for audio encoder weights (default: 'f16') */
  audioDtype: EmbeddingQuantType;

  /** Default dtype for projector weights in multimodal models (default: 'f16') */
  projectorDtype: EmbeddingQuantType;
}

/** Default quantization configuration */
export const DEFAULT_QUANTIZATION_DEFAULTS: QuantizationDefaultsSchema = {
  visionDtype: 'f16',
  audioDtype: 'f16',
  projectorDtype: 'f16',
};
