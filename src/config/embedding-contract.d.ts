import type { ManifestEmbeddingPostprocessorSchema } from './schema/manifest.schema.js';

export interface PackEmbeddingContract {
  dimension: number;
  postprocessor: ManifestEmbeddingPostprocessorSchema;
}
export function resolvePackEmbeddingContract(manifest: Record<string, any>): PackEmbeddingContract;
