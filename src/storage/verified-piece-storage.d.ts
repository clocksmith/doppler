export interface VerifiedPiece { path: string; offset: number; size: number; identity: string; }
export interface VerifiedPieceReceipt {
  indexIdentity: string; manifestIdentity: string; requestedBytes: number; verifiedBytes: number; pieces: string[];
  /** Cumulative wall time across possibly overlapping range reads; not end-to-end latency. */
  acquisitionMs: number; verificationMs: number; copyMs: number;
  /** Range output buffers owned by active reads, before transfer to the loader; not process RSS. */
  activeReadBytes: number; peakReadBytes: number;
}
export function createVerifiedPieceStorage(options: {
  manifestBytes: ArrayBuffer; indexBytes: ArrayBuffer; indexIdentity: string;
  acquire: (piece: Readonly<VerifiedPiece>, options: { signal?: AbortSignal }) => Promise<Uint8Array | ArrayBuffer>;
  signal?: AbortSignal;
}): Promise<{ manifest: import('../formats/rdrr/index.js').RDRRManifest;
  storage: ReturnType<typeof import('./artifact-storage-context.js').createArtifactStorageContext>;
  getReceipt(): VerifiedPieceReceipt }>;
