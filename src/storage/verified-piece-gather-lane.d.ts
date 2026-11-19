import type { VerifiedPieceGatherOptions, WorkerStateCachePayload, createVerifiedPieceGather } from './verified-piece-gather.js';
export function createVerifiedPieceGatherLane(options: Omit<VerifiedPieceGatherOptions, 'protectedCache'> & {
  protectedCache?: unknown;
  stateCache?: Readonly<WorkerStateCachePayload>;
  workerIndex?: number;
  workerCount?: number;
  /** The pinned index's maxPieceBytes, when the caller has already parsed that index. */
  indexMaxPieceBytes?: number;
  onStateDirty?: () => void;
}): ReturnType<typeof createVerifiedPieceGather>;
