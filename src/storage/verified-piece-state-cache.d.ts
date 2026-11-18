interface VerifiedFileStat {
  isFile(): boolean;
  dev: bigint;
  ino: bigint;
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
}

export interface FileStateSnapshot {
  device: string;
  inode: string;
  size: string;
  mtimeNs: string;
  ctimeNs: string;
}

export interface VerifiedPieceStateCacheOptions {
  /** Optional absolute directory owned by the current user; symlinks are rejected. When omitted, verification bits live in memory for the open storage session only. */
  directory?: string;
  /** Optional idle-only background piece verification on gather workers (default false). */
  backgroundVerify?: boolean;
  /** Minimum idle delay in milliseconds before running a background verification step (default 50). */
  backgroundIdleDelayMs?: number;
}

export interface WorkerFileStateDescriptor {
  canonicalPath: string;
  snapshot: FileStateSnapshot;
  wordCount: number;
  pieceCount: number;
  sharedBuffer: SharedArrayBuffer;
  stateFilePath: string | null;
}

export interface WorkerStateCachePayload {
  directory: string | null;
  backgroundVerify: boolean;
  backgroundIdleDelayMs: number;
  files: Record<string, WorkerFileStateDescriptor>;
}

export function matchesFileStateSnapshot(stat: VerifiedFileStat, snapshot: FileStateSnapshot): boolean;

export function prepareVerifiedPieceStateCache(options: {
  rawPolicy: VerifiedPieceStateCacheOptions | undefined;
  manifestIdentity: string;
  indexIdentity: string;
  index: { files: ReadonlyArray<{ path: string; size: number; pieces: Array<{ offset: number; size: number; identity: string }> }> };
  files: Record<string, string>;
}): Promise<Readonly<{
  directory: string | null;
  backgroundVerify: boolean;
  backgroundIdleDelayMs: number;
  workerPayload: Readonly<WorkerStateCachePayload>;
  checkFileStat(piecePath: string, stat: VerifiedFileStat): boolean;
  isPieceVerified(piecePath: string, offset: number): boolean;
  markPieceVerified(piecePath: string, offset: number): boolean;
  scheduleFlush(): void;
  flushSync(): void;
  close(): Promise<void>;
}> | undefined>;

export function openPieceDirectoryStateCache(options: {
  directory: string;
  pieceDirectory: string;
  manifestIdentity: string;
  indexIdentity: string;
  index: { files: ReadonlyArray<{ path: string; size: number; pieces: Array<{ offset: number; size: number; identity: string }> }> };
}): Promise<Readonly<{
  directory: string;
  isVerified(piecePath: string, offset: number, stat: VerifiedFileStat): boolean;
  markVerified(piecePath: string, offset: number, stat: VerifiedFileStat): void;
  close(): Promise<void>;
}>>;
