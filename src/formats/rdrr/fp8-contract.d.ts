interface FP8TensorRecord { dtype?: string; role?: string; layout?: string; shape?: number[]; size?: number; storage?: unknown; }
export interface FP8ManifestContract {
  quantizationInfo?: { weights?: string; compute?: string; embeddings?: string; lmHead?: string; fp8?: unknown };
  tensors?: Record<string, FP8TensorRecord>;
}
export function validateFP8ManifestContract(manifest: FP8ManifestContract): string[];
