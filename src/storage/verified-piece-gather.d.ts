export interface VerifiedPieceGatherOptions {
  manifestBytes: Uint8Array<ArrayBuffer>;
  indexBytes: Uint8Array<ArrayBuffer>;
  indexIdentity: string;
  /** Explicit approved local paths, indexed by authenticated piece-index filename. */
  files: Record<string, string>;
  maxPartBytes: number;
  /** Explicit worker count; omission preserves one lane. */
  concurrency?: number;
  /** Cancels asynchronous worker admission; does not replace read signals or close ownership. */
  startupSignal?: AbortSignal;
  /** All concurrent outputs plus two complete-piece buffers per worker; excludes metadata and native digest internals. */
  memoryBudgetBytes: number;
}
export interface VerifiedRangeLease {
  bytes: Uint8Array<ArrayBuffer>;
  /** Idempotent. Detaches bytes; call only after the consumer has copied them. */
  release(): void;
}
export function createVerifiedPieceGather(options: VerifiedPieceGatherOptions): Promise<Readonly<{
  /** At most concurrency in-flight reads or outstanding leases. Cancellation is checked before
   * publishing; close terminates the worker and rejects any pending read immediately. */
  readLease(path: string, offset: number, length: number, signal?: AbortSignal): Promise<VerifiedRangeLease>;
  /** Snapshot of worker verification evidence; requires no outstanding lease. */
  getReceipt(): Promise<import('./verified-piece-reader.js').VerifiedPieceReceipt>;
  close(): Promise<void>;
  rangeLease: VerifiedRangeLeasePort;
}>>;

export interface VerifiedRangeLeasePort {
  /** Omission denotes a legacy single-lane port. */
  concurrency?: number;
  maxPartBytes: number;
  memoryBytes: number;
  read(shard: number, offset: number, length: number, signal: AbortSignal): Promise<VerifiedRangeLease>;
}
