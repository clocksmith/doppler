export interface VerifiedPieceGatherOptions {
  manifestBytes: Uint8Array<ArrayBuffer>;
  indexBytes: Uint8Array<ArrayBuffer>;
  indexIdentity: string;
  /** Explicit approved local paths, indexed by authenticated piece-index filename. */
  files: Record<string, string>;
  maxPartBytes: number;
  /** Output plus two complete-piece buffers; excludes metadata and native digest internals. */
  memoryBudgetBytes: number;
}
export interface VerifiedRangeLease {
  bytes: Uint8Array<ArrayBuffer>;
  /** Idempotent. Detaches bytes; call only after the consumer has copied them. */
  release(): void;
}
export function createVerifiedPieceGather(options: VerifiedPieceGatherOptions): Promise<Readonly<{
  /** One in-flight read or outstanding lease per reader. Cancellation is checked before
   * publishing; close terminates the worker and rejects any pending read immediately. */
  readLease(path: string, offset: number, length: number, signal?: AbortSignal): Promise<VerifiedRangeLease>;
  close(): Promise<void>;
  rangeLease: VerifiedRangeLeasePort;
}>>;

export interface VerifiedRangeLeasePort {
  maxPartBytes: number;
  memoryBytes: number;
  read(shard: number, offset: number, length: number, signal: AbortSignal): Promise<VerifiedRangeLease>;
}
