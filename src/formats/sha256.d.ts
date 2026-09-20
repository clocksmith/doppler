/**
 * Return the SHA-256 hex digest for a UTF-8 string.
 */
export declare function sha256Hex(value: unknown): string;
export declare function sha256BytesHex(bytes: Uint8Array): string;

export interface Sha256Hasher {
  /** Consumes bytes synchronously; never retains an input view. */
  update(bytes: Uint8Array): void;
  /** Independent digest snapshot; repeated calls and later updates are supported. */
  digest(): Uint8Array;
  digestHex(): string;
}

/** Incremental SHA-256 with fixed-size workspace independent of input length. */
export declare function createSha256Hasher(): Sha256Hasher;
