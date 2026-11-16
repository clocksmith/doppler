import type { VerifiedPieceReaderOptions, VerifiedPieceReceipt } from './verified-piece-reader.js';
export type { VerifiedPiece, VerifiedPieceReceipt } from './verified-piece-reader.js';
export function createVerifiedPieceStorage(options: VerifiedPieceReaderOptions & { nodeGather?: Pick<import('./verified-piece-gather.js').VerifiedPieceGatherOptions, 'files' | 'fileIdentities' | 'maxPartBytes' | 'memoryBudgetBytes' | 'concurrency'> }): Promise<{
  manifest: import('../formats/rdrr/index.js').RDRRManifest;
  storage: ReturnType<typeof import('./artifact-storage-context.js').createArtifactStorageContext>;
  /** Generic reader evidence. Gather evidence is separate, never silently merged. */
  getReceipt(): VerifiedPieceReceipt;
  getGatherReceipt: (() => Promise<import('./verified-piece-gather.js').VerifiedPieceGatherReceipt>) | null;
}>;
