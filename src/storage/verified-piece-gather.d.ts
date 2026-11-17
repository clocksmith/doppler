import type { VerifiedPieceReceipt } from './verified-piece-reader.js';
export interface VerifiedPieceGatherOptions {
  manifestBytes: Uint8Array<ArrayBuffer>;
  indexBytes: Uint8Array<ArrayBuffer>;
  indexIdentity: string;
  /** Explicit approved regular files, indexed by authenticated piece-index filename.
   * Resolve authorized links before passing their canonical paths; leaf symlinks are rejected. */
  files: Record<string, string>;
  /** Optional explicit host snapshot. Canonical path and opened descriptor must
   * retain this device/inode/size; complete-piece authentication still applies. */
  fileIdentities?: Record<string, { device: string; inode: string; size: string }>;
  maxPartBytes: number;
  /** Explicit worker count; omission preserves one lane. */
  concurrency?: number;
  /** Cancels asynchronous worker admission; does not replace read signals or close ownership. */
  startupSignal?: AbortSignal;
  /** All outputs plus two-piece allowance per worker, used either for generic
   * piece buffers or async digest input copies. Excludes fixed native digest state, metadata and VM overhead. */
  memoryBudgetBytes: number;
}
export interface VerifiedRangeLease {
  bytes: Uint8Array<ArrayBuffer>;
  /** Idempotent. Detaches bytes; call only after the consumer has copied them. */
  release(): void;
}
export interface VerifiedPieceGatherReceipt extends VerifiedPieceReceipt {
  /** Successful full-piece reads directly into the worker-owned output, authenticated before transfer. */
  directPieceReads: number;
  directPieceBytes: number;
  /** Successfully published multi-range requests, ranges and payload bytes. */
  batchReads: number;
  batchRanges: number;
  batchBytes: number;
  /** Admitted native async SHA input copies, included in the two-piece allowance. */
  activeDigestCopyBytes: number;
  /** Per-worker peaks are summed as a conservative aggregate upper bound. */
  peakDigestCopyBytes: number;
  /** Reusable owned generic piece scratch. Never overlaps direct async digest input copies. */
  activeGenericScratchBytes: number;
  /** Per-worker peaks are summed as a conservative aggregate upper bound. */
  peakGenericScratchBytes: number;
}
export function createVerifiedPieceGather(options: VerifiedPieceGatherOptions): Promise<Readonly<{
  /** At most concurrency in-flight reads or outstanding leases. Cancellation is checked before
   * publishing; close rejects pending reads immediately and waits for settled worker ownership. */
  readLease(path: string, offset: number, length: number, signal?: AbortSignal): Promise<VerifiedRangeLease>;
  /** One concatenated owned output; one to three positive ranges share maxPartBytes. */
  readManyLease(ranges: ReadonlyArray<{ path: string; offset: number; length: number }>, signal?: AbortSignal): Promise<VerifiedRangeLease>;
  /** Snapshot of worker verification evidence; requires no outstanding lease. */
  getReceipt(): Promise<VerifiedPieceGatherReceipt>;
  /** Stops admission permanently and awaits all I/O/hash jobs and closed descriptors
   * before terminating workers. Failed retirement remains retryable.
   * Published leases remain caller-owned until release, even after close. */
  close(): Promise<void>;
  rangeLease: VerifiedRangeLeasePort;
}>>;

export interface VerifiedRangeLeasePort {
  /** True only when the worker enforces the supplied file identity snapshot. */
  fileIdentitiesBound?: boolean;
  /** readMany validates every requested path and open descriptor before reads and after authentication, before publication. */
  batchFileIdentityChecks?: 'before-read-after-auth/v1';
  /** Omission denotes a legacy single-lane port. */
  concurrency?: number;
  maxPartBytes: number;
  memoryBytes: number;
  read(shard: number, offset: number, length: number, signal: AbortSignal): Promise<VerifiedRangeLease>;
  /** Optional batch capability. One output contains ranges in caller order, without padding. */
  readMany?(ranges: ReadonlyArray<{ shard: number; offset: number; length: number }>, signal: AbortSignal): Promise<VerifiedRangeLease>;
}
