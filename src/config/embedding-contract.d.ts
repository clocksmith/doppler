import type { ManifestEmbeddingPostprocessorSchema } from './schema/manifest.schema.js';

export interface CapsuleEmbeddingContract {
  dimension: number;
  postprocessor: ManifestEmbeddingPostprocessorSchema;
}
export function resolveCapsuleEmbeddingContract(manifest: Record<string, any>): CapsuleEmbeddingContract;
