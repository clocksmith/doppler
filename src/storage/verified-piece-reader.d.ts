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
  /** Authenticates complete pieces before copying into caller-owned storage.
   * Failure may leave a verified prefix; no writes occur after settlement. */
  readInto(path: string, offset: number, destination: Uint8Array): Promise<number>;
  readBinary(path: string): Promise<ArrayBuffer>;
  /** Expected metadata from the authenticated index; never a claim that source bytes were read or verified. */
  resolveCompletePiece(path: string, offset: number, length: number): Readonly<VerifiedPiece> | null;
  /** Frozen expected full-piece descriptors intersecting a validated range; does not acquire or authenticate source bytes. */
  resolveRangePieces(path: string, offset: number, length: number): ReadonlyArray<Readonly<VerifiedPiece>>;
  getReceipt(): VerifiedPieceReceipt;
  preflight(): Promise<void>;
  close(): Promise<void>;
}>>;
