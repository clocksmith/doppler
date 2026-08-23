export interface WeightlessEmbeddingNormalization {
  type: 'rmsnorm';
  withScale: false;
  eps: number;
  position: 'after-scale';
}
export function resolveEmbeddingNormalization(value: unknown): WeightlessEmbeddingNormalization | null;
