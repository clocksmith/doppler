import type { VerifiedPieceReceipt } from './verified-piece-reader.js';
import type { VerifiedPieceStateCacheOptions, WorkerStateCachePayload, prepareVerifiedPieceStateCache } from './verified-piece-state-cache.js';
export type { VerifiedPieceStateCacheOptions, WorkerStateCachePayload } from './verified-piece-state-cache.js';
export interface VerifiedPieceProtectedCacheOptions {
  /** Explicit absolute directory (for example on tmpfs /dev/shm) owned by the current user; symlinks are rejected. */
  directory: string;
  /** Maximum bytes retained across all content-addressed piece buckets. */
  maxBytes: number;
  /** Optional host MemAvailable floor in bytes; admissions pause when freeable reserve would dip below this floor. */
  minHostAvailableBytes?: number;
  /** Optional staleness horizon in milliseconds before unhit entries yield to new workloads (default 90000). */
  staleAfterMs?: number;
}
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
  /** Optional bounded host-protected cache of authenticated pieces; every cache hit is still SHA-256 verified. */
  protectedCache?: VerifiedPieceProtectedCacheOptions;
  /** Optional per-file-state verification record cache (`(dev, ino, size, mtimeNs, ctimeNs)`) with off-critical-path persistence. */
  verificationCache?: VerifiedPieceStateCacheOptions;
  /** Internal shared state cache coordinator owned by verified-piece-storage. */
  sharedStateCache?: Awaited<ReturnType<typeof prepareVerifiedPieceStateCache>>;
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
  /** Protected cache counters; zero when protectedCache is omitted. */
  protectedCacheHits: number;
  protectedCacheHitBytes: number;
  protectedCacheMisses: number;
  protectedCacheAdmissions: number;
  protectedCacheAdmittedBytes: number;
  protectedCacheEvictions: number;
  protectedCacheEvictedBytes: number;
  protectedCacheSkips: number;
  /** Per-file-state verification cache counters; zero when verificationCache is omitted. */
  stateCacheHits: number;
  stateCacheHitBytes: number;
  stateCacheMisses: number;
  stateCacheAdmissions: number;
  stateCacheAdmittedBytes: number;
  backgroundVerifiedPieces: number;
  backgroundVerificationMs: number;
  /** Present when direct verified reads are bound: reads served to caller memory from already verified pieces. */
  directVerifiedReads?: number;
  directVerifiedBytes?: number;
  /** Direct reads declined because a piece was unverified or its file no longer matched. */
  directDeclinedReads?: number;
  /** Reads made ahead of verification, their bytes, and how many were later confirmed verified. */
  directAheadReads?: number;
  directAheadBytes?: number;
  directAheadConfirmed?: number;
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
  /** True when workers read/admit through an explicit bounded protected piece cache. */
  protectedCacheBound?: boolean;
  /** True when workers record and reuse piece verification per immutable file state. */
  verificationCacheBound?: boolean;
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
  /**
   * Optional: present when file identities and a verification cache are both bound. Fills caller memory
   * with a range whose every piece is already verified for the file's current state and resolves true;
   * resolves false (destination unusable) when a piece is unverified or the file no longer matches;
   * rejects if the file changes during the read.
   */
  readVerifiedInto?(shard: number, offset: number, destination: Uint8Array, signal?: AbortSignal): Promise<boolean>;
  /**
   * Optional, with `readVerifiedInto`. Fills caller memory before the range's pieces are verified, so the
   * read can run while a lane hashes them. Resolves true when the pieces were already verified, false when
   * the file is not the admitted one, and otherwise a check: the bytes are unauthenticated until it
   * returns true (every covering piece verified, file state unchanged), and must not be used before that.
   */
  readAheadInto?(shard: number, offset: number, destination: Uint8Array, signal?: AbortSignal): Promise<boolean | (() => boolean)>;
  /**
   * Optional: have a lane read and hash the pieces covering these ranges, waiting for a free lane instead of
   * failing when all are busy. The lane's copy of the bytes is dropped. Resolves the job's timings, and
   * rejects if a piece does not match its identity. Foreground reads are served ahead of these jobs.
   */
  verifyMany?(ranges: ReadonlyArray<{ shard: number; offset: number; length: number }>, signal?: AbortSignal): Promise<VerifiedRangeLease['timings'] | null>;
  /** Optional, with `readAheadInto`: whether every piece covering the range is already verified. No I/O. */
  hasVerified?(shard: number, offset: number, length: number): boolean;
}
