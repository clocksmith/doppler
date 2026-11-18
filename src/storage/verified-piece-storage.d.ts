import type { VerifiedPieceReaderOptions, VerifiedPieceReceipt } from './verified-piece-reader.js';
export type { VerifiedPiece, VerifiedPieceReceipt } from './verified-piece-reader.js';
export interface NodePieceDirectoryOptions {
  directory: string;
  maxPieceBytes: number;
  maxActiveReadBytes: number;
  verificationCache?: { directory: string };
}
export interface NodePieceDirectoryReceipt {
  reads: number;
  activeReadBytes: number;
  peakReadBytes: number;
  missingPieces: number;
  pendingDescriptorCloses: number;
}
export function createVerifiedPieceStorage(options: Omit<VerifiedPieceReaderOptions, 'acquire'> & {
  acquire?: VerifiedPieceReaderOptions['acquire'];
  nodeGather?: Pick<import('./verified-piece-gather.js').VerifiedPieceGatherOptions, 'files' | 'fileIdentities' | 'maxPartBytes' | 'maxRanges' | 'memoryBudgetBytes' | 'concurrency' | 'protectedCache' | 'verificationCache'>;
  nodePieceDirectory?: NodePieceDirectoryOptions;
  pieceDirectoryVerificationCache?: { directory: string; pieceDirectory: string };
}): Promise<{
  manifest: import('../formats/rdrr/index.js').RDRRManifest;
  storage: ReturnType<typeof import('./artifact-storage-context.js').createArtifactStorageContext>;
  /** Generic reader evidence. Gather evidence is separate, never silently merged. */
  getReceipt(): VerifiedPieceReceipt;
  getGatherReceipt: (() => Promise<import('./verified-piece-gather.js').VerifiedPieceGatherReceipt>) | null;
  getPieceDirectoryReceipt?: () => NodePieceDirectoryReceipt;
}>;
