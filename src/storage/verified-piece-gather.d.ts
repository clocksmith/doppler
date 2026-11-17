import type { VerifiedPieceReceipt } from './verified-piece-reader.js';
export interface VerifiedPieceGatherOptions {
  manifestBytes: Uint8Array<ArrayBuffer>;
  indexBytes: Uint8Array<ArrayBuffer>;
  indexIdentity: string;
  /** Explicit approved regular files, indexed by authenticated piece-index filename.
   * Resolve authorized links before passing their canonical paths; leaf symlinks are rejected. */
  files: Record<string, string>;
  /** Optional explicit host snapshot. Canonical path and opened descriptor must
   * retain this device/inode/size through post-authentication publication checks
   * for nonempty single and batched leases; complete-piece SHA still applies. */
  fileIdentities?: Record<string, { device: string; inode: string; size: string }>;
  maxPartBytes: number;
  /** Explicit range-count bound per output, 1..32; omission preserves three. */
  maxRanges?: number;
  /** Explicit worker count; omission preserves one lane. */
  concurrency?: number;
  /** Cancels asynchronous worker admission; does not replace read signals or close ownership. */
  startupSignal?: AbortSignal;
  /** All outputs plus a conservative two-piece allowance per worker for generic
   * piece buffers. Excludes fixed native digest state, metadata and VM overhead. */
  memoryBudgetBytes: number;
}
/** Per-job worker observations. Async read waits and different worker jobs can overlap. */
export interface VerifiedPieceJobTimings {
  workerWallMs: number;
  /** Includes native I/O and callback/scheduling wait; not isolated disk time. */
  readWaitMs: number;
  verificationMs: number;
  copyMs: number;
}
export interface VerifiedRangeLeaseTimings extends VerifiedPieceJobTimings {
  /** Lane-local elapsed time from posting the owned job until its reply, not a cross-clock subtraction. */
  roundTripMs: number;
}
export interface VerifiedRangeLease {
  bytes: Uint8Array<ArrayBuffer>;
  /** Frozen bounded scalar observations, piggybacked on the existing successful reply.
   * Omission means unsupported; zero-valued fields alone do not establish coverage. */
  timings?: Readonly<VerifiedRangeLeaseTimings>;
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
  /** Retained receipt field; direct worker-owned SHA creates no input snapshots. */
  activeDigestCopyBytes: number;
  /** Retained receipt field; zero for direct worker-owned SHA. */
  peakDigestCopyBytes: number;
  /** Reusable owned generic piece scratch. Whole-piece reads need none. */
  activeGenericScratchBytes: number;
  /** Per-worker peaks are summed as a conservative aggregate upper bound. */
  peakGenericScratchBytes: number;
}
export function createVerifiedPieceGather(options: VerifiedPieceGatherOptions): Promise<Readonly<{
  /** At most concurrency in-flight reads or outstanding leases. Cancellation is checked before
   * publishing; close rejects pending reads immediately and waits for settled worker ownership. */
  readLease(path: string, offset: number, length: number, signal?: AbortSignal): Promise<VerifiedRangeLease>;
  /** One concatenated owned output; positive ranges share maxPartBytes and the explicit maxRanges bound. */
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
  /** Omission denotes the legacy three-range batch bound. */
  maxRanges?: number;
  memoryBytes: number;
  read(shard: number, offset: number, length: number, signal: AbortSignal): Promise<VerifiedRangeLease>;
  /** Optional batch capability. One output contains ranges in caller order, without padding. */
  readMany?(ranges: ReadonlyArray<{ shard: number; offset: number; length: number }>, signal: AbortSignal): Promise<VerifiedRangeLease>;
}
