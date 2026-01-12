import type { ManifestInferenceSchema, PresetSchema, QuantizationInfoSchema } from '../config/schema/index.js';

/**
 * Detect whether model scales embeddings by sqrt(hiddenSize).
 */
export declare function detectScaleEmbeddings(
  preset: PresetSchema,
  config: Record<string, unknown>
): boolean;

/**
 * Infer embedding output layout from tensor locations.
 */
export declare function inferEmbeddingOutputConfig(
  tensorLocations: Map<string, { shape?: number[] }> | Record<string, { shape?: number[] }>
): { embeddingTranspose: boolean; embeddingVocabSize: number | null } | null;

/**
 * Build ManifestInferenceSchema from resolved preset.
 */
export declare function buildManifestInference(
  preset: PresetSchema,
  config: Record<string, unknown>,
  headDim?: number,
  quantizationInfo?: QuantizationInfoSchema | null
): ManifestInferenceSchema;
