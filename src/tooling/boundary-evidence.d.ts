export declare const SOURCE_BOUNDARY_PACK_SCHEMA: 'doppler.source-boundary-pack/v1';
export declare const RUNTIME_BOUNDARY_CAPTURE_SCHEMA: 'doppler.runtime-boundary-capture/v1';
export declare const BOUNDARY_COMPARISON_RECEIPT_SCHEMA:
  'doppler.boundary-comparison-receipt/v1';
export declare const DETERMINISTIC_TOKEN_EVIDENCE_SCHEMA:
  'doppler.deterministic-token-evidence/v1';
export declare const BOUNDARY_PROVIDER_CAPTURE_SCHEMA:
  'doppler.boundary-provider-capture/v1';

export interface BoundaryEvidence {
  boundaryId: string;
  phase: string | null;
  tokenIndex: number | null;
  layerIndex: number | null;
  occurrence?: number;
  shape: number[];
  dtype: string;
  samples: Array<{ coordinate: number[]; flatIndex: number; value: number }>;
  fullTensorDigest: string;
  statistics: Record<string, number>;
  tolerancePolicyId: string;
}

export declare function buildRuntimeBoundaryCapture(options: {
  report: unknown;
  policy: Record<string, unknown>;
  tolerancePolicyId?: string;
  identity?: Record<string, unknown>;
}): Record<string, unknown>;

export declare function buildSourceBoundaryPack(options: {
  identity: Record<string, unknown>;
  boundaries: BoundaryEvidence[];
}): Record<string, unknown>;

export declare function buildSourceBoundaryPackFromProviderCapture(
  capture: Record<string, unknown>
): Record<string, unknown>;

export declare function buildDeterministicTokenEvidenceFromReferenceTranscript(
  transcript: Record<string, unknown>
): Record<string, unknown>;

export declare function compareBoundaryEvidence(options: {
  sourcePack: Record<string, unknown>;
  runtimeCapture: Record<string, unknown>;
  policy: Record<string, unknown>;
  artifactPrecision?: 'source' | 'quantized';
  sourcePrecisionControlReceipt?: Record<string, unknown> | null;
  deterministicTokenEvidence?: { exact: boolean; tokenCount: number } | null;
}): Record<string, unknown>;
