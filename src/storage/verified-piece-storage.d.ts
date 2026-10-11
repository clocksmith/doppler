import type { VerifiedPieceReaderOptions, VerifiedPieceReceipt } from './verified-piece-reader.js';
export type { VerifiedPiece, VerifiedPieceReceipt } from './verified-piece-reader.js';
export function createVerifiedPieceStorage(options: VerifiedPieceReaderOptions): Promise<{
  manifest: import('../formats/rdrr/index.js').RDRRManifest;
  storage: ReturnType<typeof import('./artifact-storage-context.js').createArtifactStorageContext>;
  getReceipt(): VerifiedPieceReceipt;
}>;
