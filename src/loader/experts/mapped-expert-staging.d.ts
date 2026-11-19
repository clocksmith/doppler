import type { CompactLayerStats, RoutedExpertCacheOptions } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan, RoutedExpertPlane } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';

/** What one staged upload did. Times are wall-clock from the start of the upload, in milliseconds. */
export interface StagedUpload {
  /** One entry per lane job that verified ranges for the first time; empty when every range was already verified. */
  laneTimings: Array<import('../../storage/verified-piece-gather.js').VerifiedRangeLease['timings'] | null>;
  /** Bytes read into the slot while a lane was still hashing their pieces. */
  readAheadBytes: number;
  /** Until every read into the slot had settled. */
  readWaitMs: number;
  /** Until the lanes had finished; 0 when no lane ran or verification was deferred. */
  verifyWaitMs: number;
  /** True when the bytes were submitted before their verification finished, inside a release gate. */
  deferred: boolean;
}

/**
 * Open the release gate of a storage port for one step. Throws if an earlier step's deferred verification
 * failed: unverified bytes are then resident in model buffers and nothing more may be computed with them.
 */
export function beginStagedRelease(port: object): void;
/**
 * Wait for every verification deferred during the step, then close the gate. Rejects when any failed, and
 * keeps the gate failed. The step's output must not leave the stage before this resolves. With
 * `release: false` (the step already failed) it only waits and closes.
 */
export function settleStagedRelease(port: object, options?: { release?: boolean }): Promise<void>;

/** A ring of mapped upload buffers fed by direct verified reads; one per storage port. */
export interface MappedExpertStaging {
  readonly slotBytes: number;
  readonly slotCount: number;
  /** GPU bytes the ring currently owns. */
  readonly bytes: number;
  /** Resolves false when nothing was staged and nothing was submitted. */
  upload(parts: ReadonlyArray<{ shard: number; offset: number; length: number; at: number }>,
    copies: ReadonlyArray<{ at: number; length: number; target: GPUBuffer; targetOffset: number }>,
    signal: AbortSignal, beforeSubmit: (() => void) | null, stats?: RoutedExpertCounters | null, layer?: number | null): Promise<false | StagedUpload>;
  destroy(): Promise<void>;
}

/** null unless policy.uploadStaging is 'mapped', reads are batched leases, and the port serves direct verified reads. */
export function getMappedExpertStaging(options: RoutedExpertCacheOptions, plan: RoutedExpertBankPlan): MappedExpertStaging | null;
export function destroyMappedExpertStaging(options: RoutedExpertCacheOptions): Promise<void>;
/** Resolves false when the caller must load these experts through its lease path instead. */
export function stageRoutedExperts(staging: MappedExpertStaging, stats: RoutedExpertCounters, plan: RoutedExpertBankPlan,
  experts: ReadonlyArray<number>, targetOf: (expert: number, plane: RoutedExpertPlane) => { buffer: GPUBuffer; offset: number },
  signal: AbortSignal, beforeSubmit: (() => void) | null, layerStats: CompactLayerStats | null): Promise<boolean>;
