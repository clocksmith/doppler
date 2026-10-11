import type { ManifestEmbeddingPostprocessorSchema } from './schema/manifest.schema.js';

export interface CapsuleEmbeddingContract {
  dimension: number;
  postprocessor: ManifestEmbeddingPostprocessorSchema;
}
export interface EmbeddingBatchSemantics {
  readonly schema: 'doppler.embedding-batch-semantics/v1';
  readonly operation: { readonly name: 'embed'; readonly version: 1 };
  readonly inputField: 'texts';
  readonly unit: 'text';
  readonly state: 'reset-per-text';
  readonly outputOrder: 'input-order';
}
/** Supported implementation semantics only; does not qualify or authorize an application. */
export function resolveEmbeddingBatchSemantics(manifest: Record<string, unknown>): Readonly<EmbeddingBatchSemantics>;
export function resolveCapsuleEmbeddingContract(manifest: Record<string, any>): CapsuleEmbeddingContract;
