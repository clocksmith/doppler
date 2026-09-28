export interface VerifiedPiece { path: string; offset: number; size: number; identity: string; }
export function createVerifiedPieceStorage(options: {
  manifestBytes: ArrayBuffer; indexBytes: ArrayBuffer; indexIdentity: string;
  acquire: (piece: Readonly<VerifiedPiece>, options: { signal?: AbortSignal }) => Promise<Uint8Array | ArrayBuffer>;
  signal?: AbortSignal;
}): Promise<{ manifest: import('../formats/rdrr/index.js').RDRRManifest;
  storage: ReturnType<typeof import('./artifact-storage-context.js').createArtifactStorageContext>;
  getReceipt(): { indexIdentity: string; manifestIdentity: string; requestedBytes: number; verifiedBytes: number; pieces: string[] } }>;
