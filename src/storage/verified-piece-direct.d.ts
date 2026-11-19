import type { BigIntStats } from 'node:fs';

export interface VerifiedDirectReaderOptions {
  /** Index path to approved local file. */
  files: Readonly<Record<string, string>>;
  /** The admitted device, inode and size of every approved file. */
  fileIdentities: Readonly<Record<string, { device: string; inode: string; size: string }>>;
  index: { files: ReadonlyArray<{ path: string; pieces: ReadonlyArray<{ offset: number; size: number }> }> };
  stateCache: {
    checkFileStat(piecePath: string, stat: BigIntStats): boolean;
    isPieceVerified(piecePath: string, offset: number): boolean;
  };
}

export interface VerifiedDirectReader {
  /**
   * Fills `destination` from `path` at `offset` and resolves true, or resolves false without a usable
   * destination when any covering piece is unverified or the file no longer matches its verified state.
   * Rejects if the file changes during the read; the destination must then be discarded.
   */
  readInto(path: string, offset: number, destination: Uint8Array, signal?: AbortSignal): Promise<boolean>;
  /**
   * Fills `destination` whether or not its pieces are verified yet. Resolves true when they already were
   * (as `readInto`), false when the file is not the admitted one in its recorded state, and otherwise a
   * check. The destination is unauthenticated until that check returns true, which it does only once
   * every covering piece has been verified and the file is still in the state the read saw. Call it after
   * a lane has been asked to verify the same range; while it returns false the bytes must not be used.
   */
  readAheadInto(path: string, offset: number, destination: Uint8Array, signal?: AbortSignal): Promise<boolean | (() => boolean)>;
  /** True when every piece covering the range is verified for the recorded file state. No I/O. */
  isVerified(path: string, offset: number, length: number): boolean;
  getStats(): { reads: number; bytes: number; declined: number; aheadReads: number; aheadBytes: number; aheadConfirmed: number };
  close(): Promise<void>;
}

export function createVerifiedDirectReader(options: VerifiedDirectReaderOptions): VerifiedDirectReader;
