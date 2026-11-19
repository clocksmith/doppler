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
  getStats(): { reads: number; bytes: number; declined: number };
  close(): Promise<void>;
}

export function createVerifiedDirectReader(options: VerifiedDirectReaderOptions): VerifiedDirectReader;
