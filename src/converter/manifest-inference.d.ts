import type { ManifestInferenceSchema, PresetSchema } from '../config/schema/index.js';

/**
 * Detect whether model scales embeddings by sqrt(hiddenSize).
 */
export declare function detectScaleEmbeddings(
  preset: PresetSchema,
  config: Record<string, unknown>
): boolean;

/**
 * Build ManifestInferenceSchema from resolved preset.
 */
export declare function buildManifestInference(
  preset: PresetSchema,
  config: Record<string, unknown>,
  headDim?: number
): ManifestInferenceSchema;
