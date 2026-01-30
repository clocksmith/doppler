import type { ManifestInferenceSchema, PresetSchema, QuantizationInfoSchema } from '../config/schema/index.js';

/**
 * Infer embedding output layout from tensor locations.
 */
export declare function inferEmbeddingOutputConfig(
  tensorLocations: Map<string, { shape?: number[] }> | Record<string, { shape?: number[] }>
): { embeddingTranspose: boolean; embeddingVocabSize: number | null } | null;

/**
 * Build ManifestInferenceSchema from resolved preset.
 * Auto-detects normalization flags and tied embeddings from tensor names when provided.
 */
export declare function buildManifestInference(
  preset: PresetSchema,
  config: Record<string, unknown>,
  headDim?: number,
  quantizationInfo?: QuantizationInfoSchema | null,
  tensorNames?: string[] | null
): ManifestInferenceSchema;
