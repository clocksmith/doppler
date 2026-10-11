export interface VerifiedPiece { path: string; offset: number; size: number; identity: string; }
export interface VerifiedPieceReceipt {
  indexIdentity: string; manifestIdentity: string; requestedBytes: number; verifiedBytes: number; pieces: string[];
  /** Unique authenticated complete pieces actually read, bounded by the signed index. */
  ranges: VerifiedPiece[];
  /** Cumulative wall time across possibly overlapping range reads; not end-to-end latency. */
  acquisitionMs: number; verificationMs: number; copyMs: number;
  /** Range output buffers owned by active reads, before transfer to the loader; not process RSS. */
  activeReadBytes: number; peakReadBytes: number;
}
export interface VerifiedPieceReaderOptions {
  manifestBytes: ArrayBuffer | Uint8Array<ArrayBuffer>; indexBytes: ArrayBuffer | Uint8Array<ArrayBuffer>; indexIdentity: string;
  acquire: (piece: Readonly<VerifiedPiece>, options: { signal?: AbortSignal }) => Promise<Uint8Array | ArrayBuffer>;
  signal?: AbortSignal;
}
export function createVerifiedPieceReader(options: VerifiedPieceReaderOptions): Promise<Readonly<{
  manifest: import('../formats/rdrr/index.js').RDRRManifest;
  readRange(path: string, offset: number, length: number | null): Promise<ArrayBuffer>;
  readBinary(path: string): Promise<ArrayBuffer>;
  getReceipt(): VerifiedPieceReceipt;
  preflight(): Promise<void>;
  close(): Promise<void>;
}>>;
